# Design changes to disclose in Paper 2

These are implementation choices and limitations relative to the supplied proposal, not empirical findings. The original proposal remains unchanged in `source_proposal.docx`.

## World and story distribution

Use the description **“Paper 1's world, new story distribution.”** Paper 2 retains the byte-identical epistemic state engine, two characters, object/location vocabulary and forecast semantics. It does not reuse Paper 1's stories. The revised default has nine focal-object events, varied observers and a deciding observation among six interior positions. Depth-two stories follow fixed, preregistered joint quotas for whether the nested answer equals both, either or neither character's own belief. The eight-question results therefore are not directly comparable with Paper 1's eight-question panel. Report distributions and within-Paper-2 comparisons explicitly.

The decisive first-event/last-event/initial-location shortcuts were removed, but the distribution still supports weak position, location-frequency and depth-one last-joint-observation heuristics. Audit v4 scores these separately by split, depth and critical-event kind; the scores are retained even when they fail screening. The current distribution is intentionally retained rather than repeatedly redesigned until an audit looks favourable. G0 requires an explicit corpus-bound decision about these remaining risks. This is acceptance of a limitation, not a declaration that every shortcut is at chance. At depth two, the last jointly observed focal event implements the intended nested-belief update in this world; it is reported as a semantic baseline, not required to fail as a shortcut.

## Training and compute

Only story exposure is matched. SFT uses the provisional learning rate 1e-4 and 250 optimizer steps per full pool; RL uses 1e-5 and 2,000 steps, with eight completions per story. Tokens, optimizer steps and compute differ. Report unconditional results together with development-selected matched-immediate-accuracy results and missing matches. Matching does not remove every optimization confound.

RL development evaluation now occurs at steps 0, 250, …, 2,000: nine panels, 1,080 sequential native generations and nine retained evaluated adapters per run. SFT retains its 25-step interval. This reduces cost but coarsens checkpoint matching. The proposal's 15–25-hour whole-pilot estimate remains unverified. Profile training, evaluation and storage separately, account for all arms and subsequent experiments, and use the intended schedule in the cost review. See [training accounting](TRAINING.md).

## Trace interventions and precision

Twelve edited worlds are a pilot diagnostic, not the confirmatory quadrant measurement. Confirmatory preparation defaults to 97 worlds from the 120-world pool for a prespecified 95% interval half-width of 0.10 on a per-arm story-averaged rate. This is a worst-case-variance normal planning approximation; it is not a coverage guarantee, an arm-comparison power calculation, or a plan for ratios of remaining errors. Actual story-cluster intervals are also reported and checked. Changed and intact subgroups may contain fewer worlds and fail the precision criterion. Do not assign a quadrant from an imprecise axis, a small pilot or an unplanned effect threshold. Narrower precision or arm-difference claims require their own plan and potentially more worlds.

The symbolic oracle proposes a focal-belief edit and computes its eight predicted answers. A reviewer still verifies that the actual trace span expresses that fact, that the replacement changes only it, and that the paraphrase and irrelevant edits are valid. Alternative facts require an explicit reviewed prediction contract; the helper does not validate natural language automatically.

The belief-summary condition supplies the full relevant symbolic state. Its prose directly gives the answer for many questions. Interpret it as a positive control for use of a correct premise, not as evidence about the model's own summary. The second timing condition reuses the native direct-question trace across later questions; it does not introduce an untrained story-description prompt.

## Component positive controls

Full-depth K/V swaps remain meaningful primary masks. Residual masks containing layer zero, and residual masks confined to the final block, cannot support G4. Preregister a separate interior residual band or interior single-layer controls, and require measured changed-consequence effects, preservation, flat nulls and useful precision. The hook replaces block outputs: all-layer residual replacement need not exactly reproduce donor logits because other token rows can retain recipient computation. Excluding the layer-zero path is a conservative safeguard, not an assertion of exact donor equivalence.

## External benchmark

OpenToM uses adapted shared-chat prompts and option-aware scoring, with separate upstream-compatible diagnostic labels. These scores are not directly comparable with published OpenToM numbers, including Zhong et al.'s, without a separately matched protocol. See [the external-evaluation specification](OPENTOM.md).
