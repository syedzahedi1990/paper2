# Validation status

The final local suite passed **72 tests**, with zero failures or skips, on Python 3.12.14. Ruff passed, `pip check` found no broken requirements, and the Python wheel built successfully. Machine-readable evidence is in [software-validation.json](validation/software-validation.json).

Tests include genuine forward/backward and optimizer calls through all four TRL/LoRA arms on tiny random Qwen models; nonzero adapter updates in both RL arms; the 384-token cap with forced delimiters excluded from loss and retained in attention; saved/reloaded inference via the CLI; exact native component hooks and RoPE/GQA behavior; replay and regenerated traces; analysis denominators and story clustering; and failed gate/evidence checks. PEFT's ten warnings concern the deliberately in-memory synthetic test model having no pretrained model configuration file. They do not indicate skipped tests.

The regenerated corpus contains 2,000 training, 120 development and 120 confirmation worlds, verified through 17 hashed files with no world, family or rendered-story overlap. Its hash is `5328d01baf22ee99570b7bcc95807d7a5838de26adfd594a748d4cb67174ad21`. The data audit verifies 10,400 v5.5 target/forecast comparisons, exact 90% shortcut training agreement, token-length matching, and 3,120 complete held-out eight-view cue groups with identical base stories. See [data-validation.json](validation/data-validation.json).

The official OpenToM normal-length data was downloaded at commit `3f22b66276b2d7ca5fe573c28c79cc0d077aafc5`, checked with SHA-256 `11de58a9950e09d13f08df14dd595c101d4b5c183ed1ac5d0647d181c3676bf2`, and successfully imported through the external-only adapter. It is excluded from Git and reproducible with `scripts/fetch_opentom.py`.

No pretrained 3B/7B model experiment was run. GPU feasibility, learning curves on the research task, replay behavior of trained models, empirical positive controls, scientific nulls and all G0–G4 pass decisions remain to be measured during the proposed pilot. The included validation is implementation evidence, not paper results.
