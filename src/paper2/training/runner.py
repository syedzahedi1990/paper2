"""Runnable, pinned TRL 0.20 training backend. No downloads occur on import."""
from __future__ import annotations
from dataclasses import fields
from functools import partial
import importlib.metadata
import json
from pathlib import Path
import time
from typing import Any, Callable

from .config import TrainingConfig
from .data import TRACE_INFORMATION, answer_reward, make_rl_rows, make_sft_rows, rows_digest, validate_rows
from .generation import TraceCapProcessor
from .ledger import ComputeLedger
from .resources import full_pool_projection, resource_plan

PINNED = {'trl': '0.20.0', 'transformers': '4.53.2', 'peft': '0.16.0',
          'accelerate': '1.8.1', 'datasets': '3.6.0'}


def check_stack() -> dict[str, str]:
    installed = {name: importlib.metadata.version(name) for name in PINNED}
    mismatches = {name: version for name, version in installed.items() if version != PINNED[name]}
    if importlib.metadata.version('torch').split('+')[0] != '2.7.1':
        mismatches['torch'] = importlib.metadata.version('torch')
    if mismatches:
        raise RuntimeError(f'Trainer hooks require the pinned stack. Installed mismatches: {mismatches}; expected {PINNED}')
    return installed


def _sync(torch: Any, device: Any) -> None:
    """Synchronize the device doing this run, not an unrelated available GPU."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _make_trainers(config: TrainingConfig, ledger: ComputeLedger):
    import torch
    from trl import GRPOTrainer, SFTTrainer
    from transformers import LogitsProcessorList

    class TimingMixin:
        def training_step(self, *args, **kwargs):
            _sync(torch, self.model.device)
            before_generation = ledger.generation_and_scoring_seconds
            start = time.perf_counter()
            result = super().training_step(*args, **kwargs)
            _sync(torch, self.model.device)
            ledger.update_seconds += max(0, time.perf_counter() - start - (ledger.generation_and_scoring_seconds - before_generation))
            return result

        def _save_checkpoint(self, *args, **kwargs):
            _sync(torch, self.model.device)
            start = time.perf_counter()
            result = super()._save_checkpoint(*args, **kwargs)
            _sync(torch, self.model.device)
            elapsed = time.perf_counter() - start
            ledger.checkpoint_seconds += elapsed
            ledger.resumable_checkpoint_save_seconds += elapsed
            ledger.resumable_checkpoint_save_count += 1
            checkpoint = Path(self.args.output_dir) / f'checkpoint-{self.state.global_step}'
            if checkpoint.is_dir():
                snapshot = ledger.to_dict()
                elapsed = time.perf_counter() - self._paper2_run_started
                snapshot['wall_seconds'] += elapsed
                if self.model.device.type == 'cuda':
                    snapshot['gpu_seconds'] += elapsed
                    snapshot['peak_cuda_memory_bytes'] = max(snapshot['peak_cuda_memory_bytes'], int(torch.cuda.max_memory_allocated(self.model.device)))
                (checkpoint / 'paper2_compute.json').write_text(json.dumps(snapshot, indent=2) + '\n')
            return result

    class MeasuredSFTTrainer(TimingMixin, SFTTrainer):
        def compute_loss(self, model, inputs, *args, **kwargs):
            world_ids = inputs.pop('_paper2_world_ids', [])
            if model.training:
                ledger.observe_worlds(world_ids)
                count = int((inputs['labels'][..., 1:] != -100).sum().item())
                ledger.optimized_tokens += count
                ledger.supervised_tokens += count
                ledger.prompt_exposures += int(inputs['input_ids'].shape[0])
            return super().compute_loss(model, inputs, *args, **kwargs)

    class MeasuredGRPOTrainer(TimingMixin, GRPOTrainer):
        def _generate_and_score_completions(self, inputs):
            training = self.model.training
            processors = []
            original_generate = self.model.generate

            def timed_generate(*args, **kwargs):
                if config.reasoning:
                    cap = TraceCapProcessor(self.processing_class, config.trace_cap)
                    processors.append(cap)
                    existing = list(kwargs.pop('logits_processor', []))
                    kwargs['logits_processor'] = LogitsProcessorList(existing + [cap])
                _sync(torch, self.model.device)
                start = time.perf_counter()
                result = original_generate(*args, **kwargs)
                _sync(torch, self.model.device)
                if training:
                    ledger.generation_seconds += time.perf_counter() - start
                return result

            # TRL's standard single-device generation unwraps to this same model.
            # Restrict this backend to that audited generation arrangement.
            self.model.generate = timed_generate
            _sync(torch, self.model.device)
            start = time.perf_counter()
            try:
                result = super()._generate_and_score_completions(inputs)
            finally:
                self.model.generate = original_generate
            _sync(torch, self.model.device)
            if training:
                ledger.observe_worlds(row['world_id'] for row in inputs)
                ledger.generation_and_scoring_seconds += time.perf_counter() - start
                count = result['completion_ids'].shape[0]
                ledger.prompt_exposures += count // config.group_size
                ledger.completion_prompt_instances += count
                ledger.sampled_completions += count
                ledger.generated_tokens += int(result['completion_mask'].sum().item())
                eos = result['completion_ids'].eq(self.processing_class.eos_token_id)
                ledger.truncated_completions += int((~eos.any(dim=1)).sum().item())
            result['completion_attention_mask'] = result['completion_mask'].clone()
            for processor in processors:
                before = result['completion_mask']
                after = processor.mask_forced(before)
                if training:
                    ledger.forced_format_tokens += int((before - after).sum().item())
                    ledger.trace_cap_hits += len(processor.capped_rows)
                result['completion_mask'] = after
            return result

        def compute_loss(self, model, inputs, *args, **kwargs):
            if model.training:
                ledger.optimized_tokens += int(inputs['completion_mask'].sum().item())
            # Loss masking must not hide the inserted delimiter from attention.
            self._paper2_attention_mask = inputs['completion_attention_mask']
            try:
                return super().compute_loss(model, inputs, *args, **kwargs)
            finally:
                self._paper2_attention_mask = None

        def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, *args, **kwargs):
            original = getattr(self, '_paper2_attention_mask', None)
            if original is not None:
                attention_mask = torch.cat([attention_mask[:, :-logits_to_keep], original], dim=1)
            return super()._get_per_token_logps_and_entropies(model, input_ids, attention_mask, logits_to_keep, *args, **kwargs)

    return MeasuredSFTTrainer, MeasuredGRPOTrainer


def _development_predictions(model: Any, tokenizer: Any, rows: list[dict[str, Any]],
                             config: TrainingConfig, torch: Any, *,
                             formats: tuple[str, ...] = ('immediate', 'native')) -> list[dict[str, Any]]:
    """Evaluate the requested Test 1 and/or native completion/reward format.

    Native evaluation is an uninterrupted greedy completion with the training cap
    processor and correctness parser. An immediate-answer failure cannot hide a
    successful learned reasoning policy in G1.
    """
    from paper2.prompts import answer_token_ids, render_prompt
    from .data import parse_answer
    from transformers import LogitsProcessorList
    if not rows:
        raise ValueError('Development evaluation needs nonempty rows')
    if not formats or set(formats) - {'immediate', 'native'}:
        raise ValueError('Development formats must be immediate and/or native')
    records = []
    was_training = model.training
    model.eval()
    try:
        for row in rows:
            common = {key: row[key] for key in ('id', 'world_id', 'answer', 'view', 'split')}
            if 'immediate' in formats:
                prompt = render_prompt(tokenizer, row, reasoning=False)
                encoded = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
                # Match Test 1: full-vocabulary greedy next token; invalid tokens are wrong.
                # Answer labels must remain prefix-stable single tokens.
                letter_ids = answer_token_ids(tokenizer, prompt, row['choices'])
                with torch.no_grad():
                    logits = model(**encoded).logits[0, -1]
                predicted_id = int(logits.argmax().item())
                prediction = next((label for label, token in letter_ids.items() if token == predicted_id), None)
                records.append({**common, 'format': 'immediate', 'prediction': prediction,
                                'correct': prediction == row['answer'], 'token_id': predicted_id})
            if 'native' not in formats:
                continue
            native_prompt = render_prompt(tokenizer, row, reasoning=config.reasoning)
            native_input = tokenizer(native_prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
            cap = TraceCapProcessor(tokenizer, config.trace_cap) if config.reasoning else None
            budget = config.trace_cap + len(cap.delimiter_ids) + 4 if cap else 4
            with torch.inference_mode():
                generated = model.generate(**native_input, max_new_tokens=budget, do_sample=False,
                                           pad_token_id=tokenizer.pad_token_id, use_cache=True,
                                           logits_processor=LogitsProcessorList([cap]) if cap else LogitsProcessorList())
            completion = generated[0, native_input['input_ids'].shape[1]:].tolist()
            text = tokenizer.decode(completion, skip_special_tokens=True)
            native_prediction = parse_answer(text, reasoning=config.reasoning)
            native_correct = bool(answer_reward(completions=[text], answer=[row['answer']],
                                                reasoning=config.reasoning, tokenizer=tokenizer,
                                                trace_cap=config.trace_cap)[0])
            records.append({**common, 'format': 'native_reasoning' if config.reasoning else 'native_answer',
                            'prediction': native_prediction, 'correct': native_correct,
                            'completion': text, 'completion_ids': completion, 'completion_tokens': len(completion),
                            'forced_delimiter': bool(cap and cap.capped_rows),
                            'truncated': tokenizer.eos_token_id not in completion})
    finally:
        model.train(was_training)
    return records


def _upsert_jsonl(path: Path, records: list[dict[str, Any]], keys: tuple[str, ...]) -> None:
    """Resume replaces repeated checkpoint observations, never double-counting them."""
    previous = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    indexed = {tuple(record[key] for key in keys): record for record in previous + records}
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(record) + '\n' for record in indexed.values()))
    temporary.replace(path)


def run_training(config: TrainingConfig, training_rows: list[dict[str, Any]], output_dir: str | Path,
                 *, gate_file: str | Path | None = None, profile_report: str | Path | None = None,
                 dev_rows: list[dict[str, Any]] | None = None, resume_from_checkpoint: str | None = None,
                 corpus_manifest: str | Path | None = None,
                 synthetic_model: Any = None, synthetic_tokenizer: Any = None,
                 trace_builder: Callable | None = None) -> dict[str, Any]:
    """Train and save adapters, ledger and manifest.

    The only ungated entry is a supplied random tiny model (<1M parameters), used
    by the offline integration test. Published 3B/7B paths always check gates
    before model/tokenizer loading. Profile stage is an authorized pilot profile,
    still requiring frozen G0 through the gate policy.
    """
    synthetic = synthetic_model is not None
    config.validate(synthetic=synthetic)
    validate_rows(training_rows, config)
    budget = config.training_budget(len(training_rows), synthetic=synthetic)
    resolved_max_steps = budget['resolved_max_steps']
    if synthetic:
        if synthetic_tokenizer is None or sum(p.numel() for p in synthetic_model.parameters()) >= 1_000_000:
            raise ValueError('Synthetic bypass is restricted to supplied tiny models under 1M parameters')
    else:
        from paper2.gates import assert_training_allowed
        review = assert_training_allowed(config.to_dict(), gate_file, profile_report)
        from paper2.io import file_digest
        if corpus_manifest is None or file_digest(corpus_manifest) != review['calibration']['manifest_sha256']:
            raise ValueError('Training corpus manifest must match the frozen G0 manifest')
        corpus = json.loads(Path(corpus_manifest).read_text())
        if corpus['config'].get('train') != 2000 or corpus['config'].get('development') != 120 or corpus['config'].get('confirmation') != 120:
            raise ValueError('Production corpus must contain the proposal counts: 2000 train, 120 development, 120 confirmation')
        if config.stage != 'profile':
            profile = json.loads(Path(profile_report).read_text())
            if profile.get('synthetic') or profile.get('gpu_seconds', 0) <= 0:
                raise ValueError('Production training requires a real GPU profiling report')
            arrangement_keys = ('model', 'revision', 'arm', 'model_amendment_sha256', 'group_size', 'trace_cap', 'lora_rank', 'lora_alpha', 'lora_modules', 'batch_size', 'gradient_accumulation_steps', 'max_prompt_length', 'load_in_4bit', 'attn_implementation', 'bf16', 'gradient_checkpointing')
            mismatch = [key for key in arrangement_keys if profile.get('config', {}).get(key) != config.to_dict()[key]]
            if mismatch and not config.profile:
                raise ValueError(f'Training differs from the profiled arrangement: {mismatch}; reprofile before continuing')
        relative = f'supervision/{config.dataset}_reasoning_sft.jsonl' if config.arm == 'reasoning_sft' else f'training/{config.dataset}.jsonl'
        frozen_file = Path(corpus_manifest).parent / relative
        if file_digest(frozen_file) != corpus['files'][relative]['sha256']:
            raise ValueError('Training file differs from the frozen corpus manifest')
        frozen_rows = [json.loads(line) for line in frozen_file.read_text().splitlines() if line.strip()]
        if rows_digest(training_rows) != rows_digest(frozen_rows):
            raise ValueError('Supplied training rows differ from the frozen training file')
        dev_relative = 'evaluation/development__clean.jsonl'
        dev_file = Path(corpus_manifest).parent / dev_relative
        if file_digest(dev_file) != corpus['files'][dev_relative]['sha256']:
            raise ValueError('Development file differs from the frozen corpus manifest')
        frozen_dev = [json.loads(line) for line in dev_file.read_text().splitlines() if line.strip()]
        frozen_dev = [row for row in frozen_dev if row['view'] == 'direct']
        def order(row):
            return row['id']
        if not dev_rows or rows_digest(sorted(dev_rows, key=order)) != rows_digest(sorted(frozen_dev, key=order)):
            raise ValueError('Learning curves require all frozen clean development direct-view rows')
    if dev_rows is None and not synthetic:
        raise ValueError('Production runs require development rows for native and immediate learning curves')
    if dev_rows:
        train_ids = {row.get('world_id') for row in training_rows}
        for row in dev_rows:
            if row.get('split') not in ('dev', 'development') or row.get('view') != 'direct' or row.get('world_id') in train_ids:
                raise ValueError('Learning curves require disjoint development direct-view rows only')
    versions = check_stack()
    import torch
    if not synthetic and not torch.cuda.is_available():
        raise RuntimeError('Production training and profiling require a CUDA GPU; use the offline tiny-model integration tests locally')
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback, set_seed
    from trl import GRPOConfig, SFTConfig
    from accelerate import PartialState
    if PartialState(cpu=synthetic).num_processes != 1:
        raise ValueError('This profiled backend requires one process and one GPU; distributed accounting is not validated')
    set_seed(config.seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'run_manifest.json').exists() and not resume_from_checkpoint:
        raise ValueError('Output directory already contains a run; choose a new directory or resume explicitly')
    prior_ledger = None
    if resume_from_checkpoint:
        checkpoint = Path(resume_from_checkpoint).resolve()
        if checkpoint.parent != output.resolve():
            raise ValueError('Resume checkpoint must belong to this run output directory')
        previous = json.loads((output / 'run_manifest.json').read_text())
        if previous['config'] != config.to_dict() or previous['training_sha256'] != rows_digest(training_rows):
            raise ValueError('Resume config/data differ from the original run')
        prior_ledger = json.loads((checkpoint / 'paper2_compute.json').read_text())
    tokenizer = synthetic_tokenizer or AutoTokenizer.from_pretrained(config.model, revision=config.revision)
    if tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer needs an EOS token')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    delimiter_tokens = len(tokenizer.encode('\nAnswer:', add_special_tokens=False))
    planned_resources = resource_plan(config, len(training_rows), len(dev_rows or []), delimiter_tokens,
                                      synthetic=synthetic)
    evaluation_steps = set(planned_resources['evaluation_steps'])
    native_evaluation_steps = set(planned_resources['evaluation_native_steps'])
    immediate_evaluation_steps = set(planned_resources['evaluation_immediate_steps'])
    tokenizer.padding_side = 'left' if config.rl else 'right'
    if synthetic:
        model = synthetic_model
    else:
        load_kwargs = {'revision': config.revision, 'torch_dtype': torch.bfloat16 if config.bf16 else torch.float32,
                       'attn_implementation': config.attn_implementation}
        if config.load_in_4bit:
            if not torch.cuda.is_available():
                raise ValueError('4-bit training requires CUDA and bitsandbytes')
            load_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                                   bnb_4bit_compute_dtype=torch.bfloat16,
                                                                   bnb_4bit_use_double_quant=True)
            load_kwargs['device_map'] = {'': 0}
        model = AutoModelForCausalLM.from_pretrained(config.model, **load_kwargs)
        from .config import MODEL_LAYERS
        if model.config.num_hidden_layers != MODEL_LAYERS[config.model]:
            raise ValueError('Loaded model geometry differs from the immutable model profile')
    model.config.use_cache = False
    lora = LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                      bias='none', target_modules=list(config.lora_modules), task_type='CAUSAL_LM')
    ledger = ComputeLedger(training_examples=len(training_rows),
                           answer_supervision='direct-answer outcome reward' if config.rl else 'direct answer plus oracle trace' if config.reasoning else 'direct-answer label')
    if prior_ledger:
        ledger = ComputeLedger(**{field.name: prior_ledger[field.name] for field in fields(ComputeLedger) if field.name in prior_ledger})
    SFT, GRPO = _make_trainers(config, ledger)
    common = dict(output_dir=str(output), seed=config.seed, data_seed=config.seed, max_steps=resolved_max_steps,
                  per_device_train_batch_size=config.batch_size, gradient_accumulation_steps=config.gradient_accumulation_steps,
                  learning_rate=config.learning_rate, lr_scheduler_type=config.lr_scheduler_type, warmup_ratio=config.warmup_ratio,
                  logging_steps=config.logging_steps, save_steps=config.save_steps, save_total_limit=config.save_total_limit,
                  bf16=config.bf16, fp16=False, gradient_checkpointing=config.gradient_checkpointing,
                  gradient_checkpointing_kwargs={'use_reentrant': False}, report_to=[], optim='adamw_torch',
                  remove_unused_columns=False, dataloader_num_workers=0, disable_tqdm=True,
                  use_cpu=synthetic, dataloader_pin_memory=not synthetic)
    if config.rl:
        dataset_rows = make_rl_rows(training_rows, config, tokenizer)
        for row in dataset_rows:
            if len(tokenizer.encode(row['prompt'], add_special_tokens=False)) > config.max_prompt_length:
                raise ValueError('Story prompt exceeds bound; silent left truncation is prohibited')
        args = GRPOConfig(**common, num_generations=config.group_size, max_prompt_length=None,
                          max_completion_length=config.trace_cap + len(tokenizer.encode('\nAnswer:', add_special_tokens=False)) + 4 if config.reasoning else 4,
                          temperature=config.temperature, top_p=config.top_p, beta=config.beta, num_iterations=1,
                          loss_type='grpo', scale_rewards=True, mask_truncated_completions=False,
                          use_vllm=False, generation_kwargs={'use_cache': True})
        reward = partial(answer_reward, reasoning=config.reasoning, tokenizer=tokenizer, trace_cap=config.trace_cap)
        reward.__name__ = 'direct_answer_correctness'
        trainer = GRPO(model=model, processing_class=tokenizer, args=args, train_dataset=Dataset.from_list(dataset_rows),
                       reward_funcs=reward, peft_config=lora)
    else:
        dataset_rows = make_sft_rows(training_rows, config, tokenizer, trace_builder)
        args = SFTConfig(**common, max_length=None, packing=False, completion_only_loss=True,
                         dataset_kwargs={'skip_prepare_dataset': True})
        trainer = SFT(model=model, processing_class=tokenizer, args=args, train_dataset=Dataset.from_list(dataset_rows), peft_config=lora)
        ordinary_collator = trainer.data_collator
        def counted_collator(examples):
            result = ordinary_collator(examples)
            result['_paper2_world_ids'] = [example['world_id'] for example in examples]
            return result
        trainer.data_collator = counted_collator

    class RecordCurve(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            _sync(torch, trainer.model.device)
            self.optimizer_started = time.perf_counter()

        def on_optimizer_step(self, args, state, control, **kwargs):
            _sync(torch, trainer.model.device)
            elapsed = time.perf_counter() - self.optimizer_started
            ledger.optimizer_seconds += elapsed
            ledger.update_seconds += elapsed

        def on_log(self, args, state, control, logs=None, **kwargs):
            record = {'step': state.global_step, **(logs or {})}
            ledger.learning_curve.append(record)
            with (output / 'learning_curve.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')

        def evaluate(self, step):
            if not dev_rows or step not in evaluation_steps:
                return
            # Rotation may remove resumable trainer checkpoints. Preserve a small
            # adapter at every evaluated step so matching never selects missing weights.
            adapter = output / 'evaluated_checkpoints' / f'checkpoint-{step}'
            _sync(torch, trainer.model.device)
            start = time.perf_counter()
            trainer.save_model(str(adapter))
            tokenizer.save_pretrained(str(adapter))
            _sync(torch, trainer.model.device)
            elapsed = time.perf_counter() - start
            ledger.checkpoint_seconds += elapsed
            ledger.evaluated_adapter_save_seconds += elapsed
            ledger.evaluated_adapter_save_count += 1
            ledger.max_evaluated_adapter_bytes = max(ledger.max_evaluated_adapter_bytes,
                                                    sum(path.stat().st_size for path in adapter.rglob('*') if path.is_file()))
            from paper2.io import file_digest
            adapter_digest = file_digest(adapter / 'adapter_model.safetensors')
            predictions = []
            for format_name, scheduled_steps in (('immediate', immediate_evaluation_steps),
                                                 ('native', native_evaluation_steps)):
                if step not in scheduled_steps:
                    continue
                _sync(torch, trainer.model.device)
                start = time.perf_counter()
                predictions.extend(_development_predictions(trainer.model, tokenizer, dev_rows, config, torch,
                                                            formats=(format_name,)))
                _sync(torch, trainer.model.device)
                elapsed = time.perf_counter() - start
                ledger.evaluation_seconds += elapsed
                if format_name == 'native':
                    ledger.evaluation_native_seconds += elapsed
                    ledger.evaluation_native_panels += 1
                else:
                    ledger.evaluation_immediate_seconds += elapsed
                    ledger.evaluation_immediate_panels += 1
            ledger.evaluation_panels += 1
            ledger.evaluation_native_generations += sum(row['format'].startswith('native_') for row in predictions)
            ledger.evaluation_immediate_forward_calls += sum(row['format'] == 'immediate' for row in predictions)
            ledger.evaluation_generated_tokens += sum(row.get('completion_tokens', 0) for row in predictions)
            checkpoint_id = str(adapter.resolve())
            identity = {'checkpoint_id': checkpoint_id, 'path': checkpoint_id, 'checkpoint_path': checkpoint_id,
                        'checkpoint_kind': 'base' if step == 0 else 'trained',
                        'adapter_sha256': adapter_digest, 'arm': config.arm, 'dataset': config.dataset,
                        'evaluation_dataset': 'clean', 'seed': config.seed, 'split': 'development',
                        'view': 'direct', 'step': step, 'model': config.model, 'revision': config.revision}
            predictions = [{**row, **identity} for row in predictions]
            metrics = []
            for format_name in ('immediate', 'native_reasoning' if config.reasoning else 'native_answer'):
                selected = [row for row in predictions if row['format'] == format_name]
                if not selected:
                    continue
                metrics.append({**identity, 'format': format_name, 'n': len(selected),
                                'accuracy': sum(row['correct'] for row in selected) / len(selected),
                                'decoding': 'greedy_full_vocabulary',
                                'prompt_exposures': ledger.prompt_exposures,
                                'unique_worlds_seen': len(ledger.observed_world_ids)})
            _upsert_jsonl(output / 'checkpoint_evaluations.jsonl', metrics, ('checkpoint_id', 'format'))
            _upsert_jsonl(output / 'checkpoint_predictions.jsonl', predictions, ('checkpoint_id', 'format', 'id'))
            by_format = {metric['format']: metric for metric in metrics}
            native_format = 'native_reasoning' if config.reasoning else 'native_answer'
            logs = {'dev/n': len(dev_rows), 'dev/checkpoint_id': checkpoint_id,
                    'train/unique_worlds_seen': len(ledger.observed_world_ids),
                    'train/pool_coverage': len(ledger.observed_world_ids) / len(training_rows)}
            if 'immediate' in by_format:
                logs['dev/direct_immediate_accuracy'] = by_format['immediate']['accuracy']
            if native_format in by_format:
                # G1 only moves at a measured native step; never carry a stale
                # native accuracy onto a cheaper immediate-only checkpoint.
                all_metrics = [json.loads(line) for line in (output / 'checkpoint_evaluations.jsonl').read_text().splitlines()]
                baseline = next((row for row in all_metrics if row['step'] == 0 and row['format'] == native_format), None)
                latest = by_format[native_format]
                immediate = max((row for row in all_metrics if row['format'] == 'immediate' and row['step'] <= step),
                                key=lambda row: row['step'], default=None)
                g1_evidence = {'schema_version': 2, 'status': 'measurement_only_requires_gate_review',
                               'primary_format': native_format, 'baseline': baseline, 'latest': latest,
                               'native_accuracy_gain': latest['accuracy'] - baseline['accuracy'] if baseline else None,
                               'immediate_latest': immediate,
                               'checkpoint_evaluations': str((output / 'checkpoint_evaluations.jsonl').resolve()),
                               'checkpoint_predictions': str((output / 'checkpoint_predictions.jsonl').resolve()),
                               'note': 'Judge G1 in the rewarded native format; immediate-answer transfer is a separate outcome.'}
                (output / 'g1_learning_evidence.json').write_text(json.dumps(g1_evidence, indent=2) + '\n')
                logs.update({'dev/direct_native_accuracy': latest['accuracy'], 'dev/native_format': native_format})
            self.on_log(None, type('State', (), {'global_step': step})(), None, logs=logs)

        def on_train_begin(self, args, state, control, **kwargs):
            self.evaluate(state.global_step)

        def on_step_end(self, args, state, control, **kwargs):
            ledger.completed_optimizer_steps = state.global_step
            if state.global_step in evaluation_steps:
                self.evaluate(state.global_step)

    trainer.add_callback(RecordCurve())
    training_device = trainer.model.device
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    manifest = {'config': config.to_dict(), 'stack': versions, 'torch': torch.__version__,
                'training_budget': budget, 'resource_plan': planned_resources,
                'training_sha256': rows_digest(training_rows), 'training_examples': len(training_rows),
                'trainable_parameters': trainable, 'model_revision_resolved': getattr(model.config, '_commit_hash', None),
                'trace_supervision': TRACE_INFORMATION.get(config.trace_source) if config.arm == 'reasoning_sft' else None,
                'generation_arrangement': 'single process/device; transformers generate; no vLLM; standard KV cache',
                'supervision_mask': 'Completion tokens only; prompt and padding masked; injected cap delimiter masked for RL.',
                'synthetic_integration_test': synthetic, 'resume_from_checkpoint': resume_from_checkpoint,
                'training_device': str(training_device),
                'corpus_manifest': str(corpus_manifest) if corpus_manifest else None,
                'corpus_manifest_sha256': file_digest(corpus_manifest) if corpus_manifest else None,
                'gate_file': str(gate_file) if gate_file else None, 'profile_report': str(profile_report) if profile_report else None,
                'status': 'running'}
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if training_device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(training_device)
    _sync(torch, trainer.model.device)
    start = time.perf_counter()
    trainer._paper2_run_started = start
    try:
        result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        if budget['require_full_pool_coverage'] and set(ledger.observed_world_ids) != {row['world_id'] for row in training_rows}:
            raise RuntimeError('Completed optimizer budget did not cover the full training pool; inspect sampler and exposure ledger before G1')
        save_start = time.perf_counter()
        trainer.save_model(str(output / 'adapter'))
        tokenizer.save_pretrained(str(output / 'adapter'))
        trainer.save_state()
        _sync(torch, trainer.model.device)
        elapsed = time.perf_counter() - save_start
        ledger.checkpoint_seconds += elapsed
        ledger.final_adapter_save_seconds += elapsed
        manifest['train_metrics'] = result.metrics
        manifest['status'] = 'complete'
    except BaseException as error:
        manifest['status'] = 'failed'
        manifest['error'] = str(error)
        raise
    finally:
        _sync(torch, trainer.model.device)
        elapsed = time.perf_counter() - start
        ledger.wall_seconds += elapsed
        if training_device.type == 'cuda':
            ledger.gpu_seconds += elapsed
            ledger.peak_cuda_memory_bytes = max(ledger.peak_cuda_memory_bytes, int(torch.cuda.max_memory_allocated(training_device)))
        ledger.save(output / 'compute_report.json')
        (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if (output / 'g1_learning_evidence.json').is_file():
            finalize_g1_learning_evidence(output)
    report = ledger.to_dict()
    if config.profile or config.stage == 'profile':
        profile = {'status': 'completed', 'config': config.to_dict(), 'training_budget': budget,
                   'resource_plan': planned_resources, 'model': config.model,
                   'hardware': torch.cuda.get_device_name(training_device) if training_device.type == 'cuda' else 'CPU synthetic test',
                   'training_device': str(training_device),
                   'synthetic': synthetic, **report,
                   'full_pool_resource_projection': full_pool_projection(config, len(training_rows), len(dev_rows or []),
                                                                         delimiter_tokens, report, synthetic=synthetic),
                   'read_before_training': 'Review full-pool training, native development evaluation and retained snapshot storage for every arm; a short profile does not establish the pilot budget.'}
        (output / 'profile_report.json').write_text(json.dumps(profile, indent=2) + '\n')
    return {'manifest': manifest, 'compute': report, 'output_dir': str(output)}


def finalize_g1_learning_evidence(output_dir):
    """Bind every arm's final learning-curve receipt to its actual run artifacts.

    A failed or synthetic run is still recorded honestly; G1's separate integrity
    review requires completed production pilot evidence before confirmation.
    """
    from paper2.io import file_digest, write_json
    output = Path(output_dir)
    path = output / 'g1_learning_evidence.json'
    evidence = json.loads(path.read_text())
    manifest = json.loads((output / 'run_manifest.json').read_text())
    config = manifest['config']
    evidence.update({
        'schema_version': 3,
        'run_identity': {key: config[key] for key in ('arm', 'dataset', 'seed', 'model', 'revision', 'stage')},
        'run_status': manifest['status'],
        'synthetic': manifest['synthetic_integration_test'],
        'artifacts': {name: {'path': filename, 'sha256': file_digest(output / filename)}
                      for name, filename in (
                          ('manifest', 'run_manifest.json'), ('compute', 'compute_report.json'),
                          ('evaluations', 'checkpoint_evaluations.jsonl'),
                          ('predictions', 'checkpoint_predictions.jsonl'))},
        'note': 'Review each pilot arm in both its native format and immediate-answer format, including baseline and final; flat or negative gains are measurements, not automatic failures.',
    })
    write_json(path, evidence)
    return evidence
