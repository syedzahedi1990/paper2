"""Deterministic story-level splits, cue twins, provenance and leak audits."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

from .world import (AGENTS, OBJECTS, LOCATIONS, VIEWS, StoryWorld, generate_world, render_record,
                    ordinary_target, matched_pair, validate_cue_renderings, digest, build_oracle_trace)
from .provenance import UPSTREAM_PROVENANCE, SOURCE_GENERATOR_SHA256
from .shortcuts import AUDIT_VERSION, RELATION_WEIGHTS, audit_shortcuts, relation_quota

CORPUS_SCHEMA = "paper2.corpus.v3"


@dataclass(frozen=True)
class CorpusConfig:
    seed: int = 20261102
    train: int = 2000
    development: int = 120
    confirmation: int = 120
    depths: tuple[int, ...] = (1, 2)
    distractor_events: int = 2
    hidden_move: bool = False
    shortcut_agreement: float = 0.90

    def __post_init__(self) -> None:
        if any(count < 1 for count in (self.train, self.development, self.confirmation)):
            raise ValueError("Every split must contain at least one story")
        if not self.depths or any(depth not in (1, 2) for depth in self.depths):
            raise ValueError("Depths must be one and/or two")
        if self.distractor_events < 0:
            raise ValueError("distractor_events must be nonnegative")
        if not 0 <= self.shortcut_agreement <= 1:
            raise ValueError("Shortcut agreement must be a probability")
        count = self.train * self.shortcut_agreement
        if abs(round(count) - count) > 1e-8:
            raise ValueError("Training size must permit the exact requested cue agreement count")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return {"rows": count, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def generate_world_splits(config: CorpusConfig) -> dict[str, list[StoryWorld]]:
    rng = random.Random(config.seed)
    used: set[str] = set()
    result = {}
    for split in ("train", "development", "confirmation"):
        worlds = []
        object_indices = [index % len(OBJECTS) for index in range(getattr(config, split))]
        rng.shuffle(object_indices)
        relation_schedule = {}
        for kind_index in range(2):
            slots = [index for index in range(getattr(config, split))
                     if config.depths[(index // 6) % len(config.depths)] == 2
                     and (index // (6 * len(config.depths))) % 2 == kind_index]
            planned = [relation for relation, count in relation_quota(len(slots)).items() for _ in range(count)]
            rng.shuffle(planned)
            relation_schedule.update(zip(slots, planned))
        for index in range(getattr(config, split)):
            for _ in range(100):
                world = generate_world(rng, split=split, depth=config.depths[(index // 6) % len(config.depths)],
                                       event_kind=("move", "tell")[(index // (6 * len(config.depths))) % 2],
                                       object_index=object_indices[index], distractor_events=config.distractor_events,
                                       hidden_move=config.hidden_move,
                                       target_location=LOCATIONS[index % len(LOCATIONS)],
                                       deciding_position=1 + (index + index // (6 * len(config.depths) * 2)) % 6,
                                       belief_relation=relation_schedule.get(index))
                if world.family_id not in used:
                    break
            else:
                raise RuntimeError("Unable to produce a disjoint story family")
            used.add(world.family_id)
            worlds.append(world)
        rng.shuffle(worlds)
        result[split] = worlds
    return result


def _training_rows(worlds: list[StoryWorld], config: CorpusConfig) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(config.seed + 731)
    agree_indices = set(rng.sample(range(len(worlds)), round(len(worlds) * config.shortcut_agreement)))
    clean: list[dict[str, Any]] = []
    shortcut: list[dict[str, Any]] = []
    target_counts: Counter[str] = Counter()
    # Every label appears equally often (to rounding) conditional on the target.
    offsets = {target: rng.randrange(6) for target in LOCATIONS}
    for index, world in enumerate(worlds):
        target = ordinary_target(world, "direct")
        clean_cue = LOCATIONS[(target_counts[target] + offsets[target]) % 6]
        target_counts[target] += 1
        shortcut_cue = target if index in agree_indices else rng.choice([loc for loc in LOCATIONS if loc != target])
        clean.append(render_record(world, "direct", "clean", cue_location=clean_cue, dataset="clean"))
        shortcut.append(render_record(world, "direct", "shortcut", cue_location=shortcut_cue, dataset="shortcut"))
        validate_cue_renderings([clean[-1], shortcut[-1]])
    return {"clean": clean, "shortcut": shortcut}


def heldout_clean_cues(worlds: Iterable[StoryWorld]) -> dict[str, str]:
    """Preassign clean cue offsets within each depth/event-kind held-out cell.

    Every six stories use each target-relative offset exactly once. Thus a
    30-story cell has exactly five agreements without selecting stories based
    on their measured outcomes. One literal cue is reused across all views.
    """
    groups: dict[tuple[int, str], list[StoryWorld]] = defaultdict(list)
    for world in worlds:
        groups[(len(world.focal_word), world.critical_event.kind)].append(world)
    assignments = {}
    for key, cell in sorted(groups.items()):
        ordered = sorted(cell, key=lambda world: world.world_id)
        rng = random.Random(int(digest({"purpose": "heldout-clean-cue-offsets-v1", "stratum": key,
                                       "world_ids": [world.world_id for world in ordered]}), 16))
        offsets = [index % len(LOCATIONS) for index in range(len(ordered))]
        rng.shuffle(offsets)
        for world, offset in zip(ordered, offsets):
            target_index = LOCATIONS.index(ordinary_target(world, "direct"))
            assignments[world.world_id] = LOCATIONS[(target_index + offset) % len(LOCATIONS)]
    return assignments


def _evaluation_rows(worlds: list[StoryWorld]) -> dict[str, list[dict[str, Any]]]:
    outputs: dict[str, list[dict[str, Any]]] = {key: [] for key in ("clean", "agree", "independent", "conflict")}
    clean_cues = heldout_clean_cues(worlds)
    for world in worlds:
        target = ordinary_target(world, "direct")
        for view in VIEWS:
            variants = [render_record(world, view, "clean", cue_location=clean_cues[world.world_id]),
                        render_record(world, view, "agree")]
            variants += [render_record(world, view, "independent", cue_location=loc) for loc in LOCATIONS]
            variants += [render_record(world, view, "conflict", cue_location=loc) for loc in LOCATIONS if loc != target]
            validate_cue_renderings(variants)
            for row in variants:
                outputs[row["cue_condition"]].append(row)
    return outputs


def validate_splits(splits: Mapping[str, Iterable[StoryWorld]]) -> dict[str, Any]:
    family_owner: dict[str, str] = {}
    world_owner: dict[str, str] = {}
    story_owner: dict[str, str] = {}
    counts = {}
    for split, worlds in splits.items():
        count = 0
        for world in worlds:
            if world.split != split:
                raise ValueError("World's split does not match its containing split")
            for key, registry in ((world.family_id, family_owner), (world.world_id, world_owner),
                                  (digest(render_record(world, "direct")["story"]), story_owner)):
                if key in registry:
                    raise ValueError(f"Story leakage or duplicate between {registry[key]} and {split}")
                registry[key] = split
            count += 1
        counts[split] = count
    return {"story_counts": counts, "world_overlap": 0, "family_overlap": 0, "rendered_story_overlap": 0}


def validate_training_rows(rows: Iterable[Mapping[str, Any]], *, allow_trace: bool = False) -> None:
    seen: set[str] = set()
    for row in rows:
        if row.get("split") != "train" or row.get("view") != "direct":
            raise ValueError("Only direct-belief questions from training stories may be optimised")
        if row["world_id"] in seen:
            raise ValueError("A training story occurs more than once")
        seen.add(row["world_id"])
        if any(key in row for key in ("world", "forecasts", "final_state", "consequence_targets")):
            raise ValueError("Training export contains unrewarded targets or symbolic state")
        if not allow_trace and any(key in row for key in ("trace", "oracle_trace")):
            raise ValueError("Label-only / outcome-RL training export contains trace supervision")
        if row["answer"] not in row["choices"]:
            raise ValueError("Training answer is outside the single-letter codebook")


def generate_corpus(output_dir: str | Path, config: CorpusConfig | None = None, **kwargs: Any) -> dict[str, Any]:
    """Materialise a new frozen corpus; refuse to overwrite an existing manifest.

    `kwargs` are CorpusConfig fields for convenient library / CLI use. Held-out
    independent and conflict variants share a story ID; analyses must first
    average the six/five replicas within each world and consequence view.
    """
    if config is not None and kwargs:
        raise ValueError("Use either a CorpusConfig or keyword configuration")
    config = config or CorpusConfig(**kwargs)
    output = Path(output_dir)
    if (output / "manifest.json").exists():
        raise FileExistsError("Frozen corpus already exists; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    splits = generate_world_splits(config)
    leak_audit = validate_splits(splits)
    files = {}
    for split, worlds in splits.items():
        name = f"worlds/{split}.jsonl"
        files[name] = _write_jsonl(output / name, [world.to_dict() for world in worlds])
    training = _training_rows(splits["train"], config)
    shortcut_audit = {"train": audit_shortcuts(splits["train"], cue_rows=training["clean"],
                                                expected_depths=config.depths, require_cues=True, require_relation_quotas=True)}
    for split in ("development", "confirmation"):
        clean_cues = heldout_clean_cues(splits[split])
        shortcut_audit[split] = audit_shortcuts(splits[split], cue_rows=[render_record(world, "direct", "clean",
                                                                                   cue_location=clean_cues[world.world_id])
                                                                     for world in splits[split]],
                                                expected_depths=config.depths, require_cues=True, require_relation_quotas=True,
                                                require_clean_cue_quota=True)
    for dataset, rows in training.items():
        labels, reasoning = [], []
        for row in rows:
            label = {key: value for key, value in row.items() if key != "world"}
            labels.append(label)
            reasoning.append({**label, "trace": build_oracle_trace(row),
                              "trace_source": "paper1_symbolic_oracle_v1",
                              "trace_information": "Explicit task update rule and intermediate focal-belief states"})
        validate_training_rows(labels)
        validate_training_rows(reasoning, allow_trace=True)
        name = f"training/{dataset}.jsonl"
        files[name] = _write_jsonl(output / name, labels)
        name = f"supervision/{dataset}_reasoning_sft.jsonl"
        files[name] = _write_jsonl(output / name, reasoning)
    for split in ("development", "confirmation"):
        evaluation = _evaluation_rows(splits[split])
        for condition, rows in evaluation.items():
            name = f"evaluation/{split}__{condition}.jsonl"
            files[name] = _write_jsonl(output / name, rows)
        name = f"pairs/{split}.jsonl"
        clean_cues = heldout_clean_cues(splits[split])
        files[name] = _write_jsonl(output / name, [matched_pair(world, cue_location=clean_cues[world.world_id])
                                                for world in splits[split]])
    source_path = Path(__file__).parent / "_vendor" / "tom_epistemic_generator.py"
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != SOURCE_GENERATOR_SHA256:
        raise ValueError("Vendored Paper 1 state engine differs from its pinned source")
    implementation = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted(Path(__file__).parent.glob("*.py"))}
    manifest = {"schema_version": CORPUS_SCHEMA, "config": asdict(config),
                "shortcut_audit": shortcut_audit,
                "integrity_passed": all(audit["integrity_passed"] for audit in shortcut_audit.values()),
                "empirical_calibration_status": "requires_named_review",
                "depth_two_relation_design": {"weights_out_of_36": RELATION_WEIGHTS,
                                               "allocation": "Largest remainder, separately in every split and decisive-event kind",
                                               "assignment": "Quota labels shuffled before world generation; no outcome-based rejection sampling"},
                "world": {"objects": list(OBJECTS), "locations": list(LOCATIONS), "agents": list(AGENTS),
                          "max_belief_depth": 2, "views": list(VIEWS)},
                "training_views": ["direct"], "unrewarded_views": list(VIEWS[1:]),
                "source_generator_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "upstream_provenance": UPSTREAM_PROVENANCE,
                "implementation_sha256": implementation,
                "split_audit": leak_audit, "files": files,
                "cue_design": {"shortcut_training_agreement": sum(row["cue_agrees"] for row in training["shortcut"]) / config.train,
                               "clean_training_agreement": sum(row["cue_agrees"] for row in training["clean"]) / config.train,
                               "heldout_clean_assignment": "Target-relative cue offsets counterbalanced within each depth/event-kind cell; same cue reused for paired swaps",
                               "agree": 1.0, "independent": 1 / 6, "conflict": 0.0,
                               "heldout_variants_per_world_view": {"clean": 1, "agree": 1, "independent": 6, "conflict": 5},
                               "agreement_reference": "Rewarded direct-belief target of the underlying world, fixed across all eight views",
                               "same_inventory_order_across_views": True,
                               "matching": "Exact character/word length and entity frequencies; tokenizer audit separately required",
                               "analysis_unit": "Average cue variants within world/view, then aggregate stories",
                               "joint_group_by": ["world_id", "cue_condition", "cue_location"]},
                "reasoning_sft_trace_source": "paper1_symbolic_oracle_v1",
                "depth_two_extension": "Nested direct/search/report projection; all targets computed by unchanged Paper 1 state engine",
                "calibration": "Freeze this manifest after development calibration and before any training",
                "corpus_hash": digest({"schema_version": CORPUS_SCHEMA, "config": asdict(config),
                                       "files": files, "shortcut_audit": shortcut_audit})}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def verify_corpus(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("schema_version") != CORPUS_SCHEMA:
        raise ValueError("Obsolete corpus schema; regenerate v3 data and recalibrate G0")
    if any(audit.get("audit_version") != AUDIT_VERSION for audit in manifest.get("shortcut_audit", {}).values()):
        raise ValueError("Obsolete shortcut audit; regenerate with audit v5 and recalibrate G0")
    expected_hash = digest({"schema_version": CORPUS_SCHEMA, "config": manifest["config"],
                            "files": manifest["files"], "shortcut_audit": manifest["shortcut_audit"]})
    if manifest.get("corpus_hash") != expected_hash:
        raise ValueError("Frozen corpus manifest hash mismatch")
    for name, info in manifest["files"].items():
        path = output / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Frozen dataset checksum mismatch: {name}")
        if len(read_jsonl(path)) != info["rows"]:
            raise ValueError(f"Frozen dataset row count mismatch: {name}")
    splits = {split: [StoryWorld.from_dict(row) for row in read_jsonl(output / f"worlds/{split}.jsonl")]
              for split in ("train", "development", "confirmation")}
    audit = validate_splits(splits)
    if audit != manifest["split_audit"]:
        raise ValueError("Frozen split audit changed")
    for dataset in ("clean", "shortcut"):
        validate_training_rows(read_jsonl(output / f"training/{dataset}.jsonl"))
        validate_training_rows(read_jsonl(output / f"supervision/{dataset}_reasoning_sft.jsonl"), allow_trace=True)
    shortcut_audit = {}
    for split, worlds in splits.items():
        clean_rows = read_jsonl(output / ("training/clean.jsonl" if split == "train"
                                         else f"evaluation/{split}__clean.jsonl"))
        shortcut_audit[split] = audit_shortcuts(worlds, cue_rows=clean_rows,
                                                expected_depths=manifest["config"]["depths"], require_cues=True,
                                                require_relation_quotas=True, require_clean_cue_quota=split != "train")
        if split != "train":
            planned_cues = heldout_clean_cues(worlds)
            if any(row["cue_location"] != planned_cues[row["world_id"]] for row in clean_rows):
                raise ValueError("Held-out clean cue assignment differs from its counterbalanced plan")
            if any(pair.get("cue_location") != planned_cues[pair["world_id"]]
                   for pair in read_jsonl(output / f"pairs/{split}.jsonl")):
                raise ValueError("Matched-pair cue differs from the clean evaluation cue")
    if shortcut_audit != manifest["shortcut_audit"]:
        raise ValueError("Frozen shortcut audit changed")
    integrity_passed = all(result["integrity_passed"] for result in shortcut_audit.values())
    if (manifest.get("integrity_passed") != integrity_passed or
            manifest.get("empirical_calibration_status") != "requires_named_review"):
        raise ValueError("Frozen corpus integrity or empirical-review status changed")
    return {"verified": True, "files": len(manifest["files"]), "corpus_hash": manifest["corpus_hash"],
            "integrity_passed": integrity_passed, "empirical_calibration_status": "requires_named_review",
            "shortcut_audit": shortcut_audit, **audit}


def assert_calibration_corpus(output_dir: str | Path) -> dict[str, Any]:
    """Recheck fixed corpus design and provenance without grading heuristics."""
    output = Path(output_dir)
    result = verify_corpus(output.parent if output.name == "manifest.json" else output)
    if not result["integrity_passed"]:
        raise ValueError("G0 requires intact declared quotas and complete stratified corpus coverage")
    if result["story_counts"] != {"train": 2000, "development": 120, "confirmation": 120}:
        raise ValueError("G0 calibration uses the proposal's fixed 2000/120/120 story splits")
    if any(audit["audit_version"] != AUDIT_VERSION for audit in result["shortcut_audit"].values()):
        raise ValueError("Obsolete shortcut audit; regenerate and recalibrate G0")
    return result


def audit_tokenizer_lengths(rows: Iterable[Mapping[str, Any]], tokenizer: Any) -> dict[str, Any]:
    """G2 tokenizer-specific guard. Fails closed if a model breaks length matching."""
    groups: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in rows:
        encoded = tokenizer.encode(row["story"], add_special_tokens=False)
        groups[(row["world_id"], row["view"])].add(len(encoded))
    failures = [list(key) for key, lengths in groups.items() if len(lengths) != 1]
    return {"passed": not failures, "groups": len(groups), "mismatched_groups": failures,
            "tokenizer": getattr(tokenizer, "name_or_path", type(tokenizer).__name__)}
