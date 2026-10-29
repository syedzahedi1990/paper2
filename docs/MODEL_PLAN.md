# Capability-based model amendment

The default remains Qwen2.5-3B for the pilot and Qwen2.5-7B for confirmation. Supporting an alternative does not select it or approve a gate. A named reviewer can instead freeze a two-stage model plan at G0, based on completed development capability screens and a written rationale. No accuracy cutoff is inferred by software.

Set `settings.model_plan` in the calibration settings to:

```json
{
  "pilot": {"model": "<selected model ID>", "revision": "<immutable revision>"},
  "confirmatory": {"model": "<selected model ID>", "revision": "<immutable revision>"},
  "amendment": {"path": "model-amendment.json", "sha256": "<file SHA256>"}
}
```

The amendment path is relative to the gate-review file. The amendment itself is a JSON document with `schema_version: 1`, `kind: "model_capability_amendment"`, `model_plan` containing the same `pilot` and `confirmatory` identities, `manifest_sha256`, a named `reviewed_by`, a nonempty `rationale`, and `capability_screens`. Each screen lists `protocol`, `summary` and `worlds` objects, each with `path` and `sha256`; these paths are relative to the amendment. Use the immutable outputs from `scripts/run_capability_screen.py`, including `worlds.jsonl`. The evidence must cover the original 3B pilot and both selected identities, using the same 120 clean development stories, all eight immediate views, and the direct native-reasoning view. Validation binds the actual frozen development file, every story/question ID and answer target, canonical protocol and record digests, and common script/implementation hashes across screens. External file hashes remain hashes of bytes. Rejected or weaker screens remain evidence; no confirmation outcomes select the models.

Supported profiles are pinned explicitly:

| Model | Revision | Layers | Primary Test 3 band |
|---|---|---:|---|
| Qwen/Qwen2.5-3B-Instruct | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | 36 | 1–35 |
| Qwen/Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` | 28 | 1–27 |
| mistralai/Mistral-Small-24B-Instruct-2501 | `9527884be6e5616bdd54de542f9ae13384489724` | 40 | 1–39 |

The loader checks the layer count against the pinned profile. `g4_layer_policy` remains `all_after_embedding_layer`; the frozen `g4_test_layers` must cover the declared pilot's complete band. Each confirmatory model uses its own complete band.

Every training/profile config in an amended lineage must set `model_amendment_sha256` to the amendment's hash and match its stage's exact model/revision, including a stage whose model remains the default. Run `paper2 freeze-calibration` normally: it verifies the amendment and screen hashes and adds the amendment to G0's reviewed evidence. Later gate checks repeat the bindings. Editing the declaration without corresponding reviewed evidence is rejected. Use a new lineage to amend an already frozen calibration.

An initial pilot profile (`stage: "profile"`) requires the frozen G0, with no previous profile; its model is determined by the declared pilot identity. Every production profile requires an explicit positive `max_steps` budget, so omission cannot silently launch a full training-pool pass. Each full pilot/confirmatory run then needs its own arm's measured profile with the same model, revision and amendment hash. G1 still requires all three complete clean seed-1 pilot arms, full training-pool exposure, and complete native/immediate development curves. `paper2 verify-g1` obtains the pilot identity from the same calibration.

All confirmatory runs, including profiling, retain G0–G4 and the separate cost approval under the £500 ceiling. For an initial target-model profile only, set all three fields `stage: "confirmatory"`, `profile: true`, and `target_profile_bootstrap: true`. This may cite a completed profile of the calibrated pilot model, including a different pilot arm when profiling reasoning SFT. It does not authorize a full run. Subsequent full runs require their own exact model/revision/arm profile. Profiling adapters are disposable; learning arms restart from the original base checkpoint.
