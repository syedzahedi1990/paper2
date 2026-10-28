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
its word observes or receives the event. Initial locations are common knowledge.

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
location. Their event-rewrite targets are obtained by replaying the original
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
the depth mixture, irrelevant-event count and an optional later hidden move;
calibration must not consult confirmation stories. Re-running generation into a
directory containing a manifest is refused. `verify_corpus` validates file
hashes, row counts, story splits and direct-only training contracts.
