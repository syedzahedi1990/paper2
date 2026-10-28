# Paper 2 task data

The vendored `tom_epistemic_generator.py` is byte-identical to the owner's v5.5
experiment bundle. `provenance.py` pins that file and records the related public
release, the eight-view definitions and the archived reference dataset. The
public release and local full-generator bundle are identified separately; the
local generator is not misrepresented as a file fetched from the public commit.
No distribution license has been invented.

The world has the original twenty objects, six locations, Alice and Bob, and all
belief coordinates of depth at most two. A move changes the physical location;
a tell does not. A belief coordinate changes precisely when every character in
its word observes or receives the event. Initial locations are common knowledge. The **state engine** is unchanged; the
v3 sampling distribution is a deliberate Paper 2 extension, not a claim to
reproduce Paper 1's short narrative distribution.

Each story now has nine relevant events about the focal object, plus the
configured irrelevant-object events. One of six interior positions is the last
event received by every character in the queried belief. Earlier events may or
may not be observed, and later relevant events do not reach all those
characters. The decisive event is never marked in narrative text. Its location
is balanced across the six answer labels; other event locations and the
initial location are independent draws, except for the final depth-two private
observations that realize the declared relation quotas below. First-event, last-event and
initial-location lookup therefore have a one-sixth population success rate.
That claim is limited to those three rules: interior event positions and
location-frequency rules remain informative, as measured below. This is
**Paper 1's world, new story distribution**. The eight-view scores are not
directly comparable to Paper 1's eight-question panel, despite sharing its
state engine and first-order view roles.
At depth two, the final private observations separately update the two
characters, in randomized order, to locations realizing the independent
six-label **joint relation quotas**. Before any world is generated, each
split × event-kind cell is assigned integer counts for `both`, `outer_only`,
`inner_only`, and `neither`, using largest-remainder rounding of weights
1:5:5:25. These labels are shuffled across the cell's preassigned world slots.
Thus each 30-world held-out cell has counts 1, 4, 4, 21. Each 500-world cell
has 14, 70, 69, 347; other sizes use the same fixed formula. Equal final
beliefs use the target location; unequal beliefs independently sample one of
the other five locations. This balances the relation in finite samples without
selecting or rejecting worlds based on measured shortcut scores. Hidden moves
are placed before the final private observations are resolved, so the planned
relation survives that difficulty setting. The nested answer
can equal both characters' own beliefs, one, or neither. Equality is **not**
forbidden to manufacture a depth-two difficulty condition. Extra hidden moves
have random locations and random positions after the deciding observation;
they are not invariably the final sentence or invariably a false-belief case.

The eight ordinary views are `direct`, `world`, `other_agent`, `search`,
`observed_overwrite`, `unobserved_overwrite`, `irrelevant_object`, and `report`.
Their targets are evaluated by the original symbolic engine. Answer labels A–F
map to the six location names, allowing a single generated answer token without
assuming each location spelling is itself a single model token. Model-specific
token checks still run before experiments.

First-order view roles match v5.5. For second-order records, `direct` asks what
one character believes the other believes, `search` asks where the first expects
the second to search, and `report` communicates the queried nested belief. An
observed overwrite is witnessed by both characters; an unobserved overwrite is
witnessed only by the inner character. This rendering extension is explicitly
tagged `nested-consequence-projections-v1` in each second-order record. A natural
event rewrite can change the other character's answer at depth two; the per-pair
change mask therefore comes from the oracle, not from fixed view names. The
belief-edit dependent views remain direct/search/unobserved-overwrite/report.
Records compute `belief_edit_change` and `event_rewrite_change` separately from
the oracle, and pairs carry both masks. `target_scope` refers specifically to
the focal-coordinate belief edit, never to an event rewrite. Multiple later
observations can erase some event-rewrite effects; account forecasts can
coincide in a particular world, which must not be scored as failed preservation.

Default generation creates 2,000 training, 120 development and 120 confirmation
worlds. Depth, event kind and target label are factorially balanced to rounding;
object identities are independently shuffled and balanced to rounding. World
families, worlds and rendered base narratives cannot cross splits. A family
collapses critical-event-value variants so that natural donors cannot leak
between splits. Training exports contain only the direct question and its
ordinary answer, with no world state or other consequence labels.

Every cue-bearing rendering appends six explicitly irrelevant inventory-label
sentences, one per location. Changing their order preserves character length,
word length and the full entity-frequency vector. The last-mentioned story
location is the planted cue. The clean training set balances this cue conditional
on the true answer; its shortcut twin changes only the inventory order and has
exactly 90% agreement. Thus the two training datasets use the same worlds and the
same amount of text, and the clean set lacks the planted answer correlation.

Held-out **clean** cues use target-relative offsets 0–5 counterbalanced before
rendering within each depth × decisive-event-kind cell. A 30-world cell has
exactly five of each offset, hence exactly five agreements; any remainder is
allocated in fixed offset order. The offsets are deterministically shuffled
across world IDs. This is an ex-ante design constraint, not outcome-driven
resampling. All eight views and both sides of each Test 3 pair reuse the same
literal clean cue and inventory order. Corpus verification recomputes the
assignment and checks the exact per-cell agreement quota. The shortcut audit
version is v4; older audits cannot authorize G0.

Held-out cue conditions refer to the **rewarded direct-belief target of the
underlying world**, not to each later consequence answer. The inventory order is
fixed across all eight views of a given world/cue rendering. Conditions use one
agreeing rendering, six independent renderings (all locations), and five
conflicting renderings (all locations other than the direct-belief target).
Thus a cue agreeing with the rewarded answer can disagree with an overwrite or
irrelevant-object answer; a conflicting cue can agree with such a consequence.
`cue_agrees_direct` records the designed relation, `cue_agrees_evaluated` records
the relation to the current ordinary target, and `cue_agrees` aliases the direct
relation. Consequence questions never control cue assignment.

The independent condition has exactly one-sixth agreement **within every
world**, and the same six inventory orders are reused across its eight views.
Aggregate replicas within world/view before computing story-level comparisons
and bootstrap intervals. For joint eight-view correctness, group by
`(world_id, cue_condition, cue_location)`; each group has all eight views and one
identical `base_story`. `id` / `rendering_id` identify the exact variant;
`world_id` identifies the shared world. The manifest reports all realised
training cue rates against the rewarded target. Tokenizer-specific length equality is checked
separately by `audit_tokenizer_lengths`; textual matching does not pretend to
prove equality under an arbitrary tokenizer.

`base_story` omits later consequence events and questions for the pre-question
trace condition. It preserves the current inventory-label order. Full evaluation
stories then add the relevant consequence event. The `critical_event_span` and
`critical_value_span` are half-open character offsets into `story`; prompt
construction must translate them before tokenization.

Test 3 pairs differ in exactly one critical observation/communication event's
location. Both sides include the training-format inventory appendix, held byte-for-byte
fixed across the pair and all views, so the only textual event difference is
the critical location. Their event-rewrite targets are obtained by replaying the original
state engine with that one change. Other named account forecasts are separate
from ordinary targets and are used only in the specified swap analysis. A tell
rewrite does not change physical-world state. No rewrite forecast is used for
ordinary Test 1 or Test 2 correctness.

Reasoning-SFT supervision is supplied in separate `supervision/` files. Its
stated source is `paper1_symbolic_oracle_v1`, exposing the transition rule and
intermediate focal-belief states. This is additional information relative to
label-only SFT and outcome-reward RL and must be reported as such. The other
training files do not contain traces.

Every export is deterministic and content-hashed. Freeze the corpus manifest
after development calibration and before training. Difficulty parameters include
the depth mixture, irrelevant-event count and an optional hidden move;
calibration must not consult confirmation stories. Re-running generation into a
directory containing a manifest is refused. `verify_corpus` validates file
hashes, row counts, story splits, direct-only training contracts and the entire
recomputed shortcut audit. The corpus/world schema remains v3 because this
review does not alter story sampling. Audit v4 changes the recorded screening
results and corpus hash: old manifests must be regenerated and recalibrated.

`audit_shortcuts` reports each split separately and each depth × decisive-event
kind separately. Its required controls screen first-event, last-event, initial-location and clean
cue rules against one-sixth; planted cue conditions retain their intended
90%/100%/0% rates. For depth two it reports equality with either own belief,
equality with both, and disagreement with both, plus the expected answer
accuracy of choosing uniformly among labels excluding one or both own beliefs.
The independent six-label nulls are respectively 1/6, 1/36 and 25/36 for these
relations. In particular, disagreement with both is naturally common and must
not be tested against a 50% null. The exclusion rules themselves have a 1/6
accuracy null. Thus both an always-equal generator and an always-different
generator fail the audit, without imposing the opposite shortcut.

The manifest stores every score, sample size and explicit simultaneous
Hoeffding screening threshold (familywise alpha 0.01 over 52 required-control comparisons).
The per-character equality and different-from-both relations must also exhibit
both outcomes in every sufficiently populated depth-two cell, even when a
small-sample concentration bound alone would admit a deterministic relation.
Equality with both is exempt from the generic nondegeneracy guard because its
1/36 null can give zero cases in unconditioned samples. Corpus readiness uses a
stronger guard: the observed four-way joint counts must **exactly** equal the
preassigned integer quota in every split/event-kind cell. Planned relation
metrics have a maximum deviation of one rounding unit per joint cell (two for
a marginal), rather than the broad sampling bound. For example, a 30-world
cell with 28/30 `different_from_both` cases cannot pass. Missing configured
depth/event cells or missing clean cues fail corpus readiness. No automatic
resampling hides a failed audit; the relation allocation is a declared design
constraint applied uniformly to training, development and confirmation. These are prespecified checks, not significance claims or substitutes for G0
model calibration. At least 24 worlds are required per observed stratum; tiny
smoke corpora can be generated and checksum-verified but are marked
`calibration_ready: false`. `assert_calibration_corpus` verifies the actual
corpus files and recomputes the audit before G0, rather than trusting a manually
edited pass flag. Default training has about 500 worlds per stratum; the
120-world held-out sets have 30, so their finite-sample bounds are explicitly
wider. A failure is not repaired by looking at confirmation model results.

Audit v4 additionally measures every existing **focal-object event position**,
numbered from one, including positions outside the six possible deciding slots.
If a position is absent in a variable-length story, its preregistered fallback
is the focal object's initial location; such cases remain in the denominator
and their counts are reported. Three frequency rules take the modal location
over (1) focal-object events, (2) those events plus its initial location, and
(3) all objects' initial locations and all story events. The inventory appendix
adds one occurrence of every location, so cannot change the third rule's mode.
All frequency ties score the exact accuracy of uniform choice among tied
locations; they are never resolved using the answer. The audit also evaluates
the last focal-object event seen by both characters, with initial location as
its fallback. This is a shortcut at depth one. At depth two it is exactly the
inherited engine's correct update rule, so its accuracy is reported separately
as a **semantic baseline**, not compared against a chance null.

These new rules expose remaining weak heuristics. On the default 2,000 training
worlds, fixed positions 2–7 score 27.7–33.5% within depth/event-kind cells; the
focal-event mode scores 33.5–38.2%; and last-both-observed scores 56.5–58.0% at
depth one. Chance is 16.7%. Depth-two last-both-observed scores 100%, as the
state rule predicts. The audit preserves the story distribution and reports
these limitations; it does not reroll stories until the new rules pass.
All scores, tie/fallback counts and failures appear in each stratum's
`residual_heuristics` section. These screens use a separate fixed family of
four strata × (all observed positions + four additional rules), with a minimum
of 52 comparisons; the required controls' thresholds are unchanged. Screening
on a small held-out cell can miss a heuristic that is detectable in training,
so a held-out pass is not evidence that the rule lacks predictive information.

The default corpus therefore has `required_controls_passed: true`,
`requires_heuristic_review: true`, and `calibration_ready: false`.
Top-level audit `passed` and `all_shortcuts_near_chance` remain false. G0 must
explicitly accept and disclose the measured remaining heuristics, using a
named review with a rationale bound to the current corpus hash. Acknowledgement
does not rewrite the failed screen or waive missing strata, insufficient
sample sizes, the original required controls, exact quotas, or file integrity.
The paper must state that rewarded accuracy can improve through these weak
heuristics; the unrewarded consequence and intervention results must establish
the claimed belief-tracking change. This review is a prerequisite for empirical
calibration, not evidence that G0 or any other empirical gate has passed.
