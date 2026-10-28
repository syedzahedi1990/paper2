"""Paper 1's finite epistemic world, with eight ordinary consequence views.

The state transition function is imported unchanged from the v5.5 source. At
depth two, the eight view *roles* project the queried nested belief; this small
rendering extension is explicit in records rather than silently collapsing it
to first order. In particular, an observer-other-agent target need not be an
invariant when both characters observed the critical event.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import random
import re
from collections import Counter
from typing import Any, Mapping

from ._vendor.tom_epistemic_generator import (
    EpistemicState, Event, TaskConfig, verbalize_event, verbalize_initial,
)

AGENTS = ("Alice", "Bob")
OBJECTS = ("key", "book", "coin", "cup", "phone", "map", "ring", "ticket", "card", "pen",
           "toy", "watch", "badge", "bottle", "notebook", "wallet", "scarf", "photo", "candle", "letter")
LOCATIONS = ("box", "basket", "shelf", "drawer", "cabinet", "closet")
CHOICES = dict(zip("ABCDEF", LOCATIONS))
VIEWS = ("direct", "world", "other_agent", "search", "observed_overwrite",
         "unobserved_overwrite", "irrelevant_object", "report")
ACCOUNTS = ("belief_edit", "event_rewrite", "broad_state_write", "persistent_direct_override",
            "cleared_direct_override", "global_output_permutation")
CONFIG = TaskConfig(agents=AGENTS, objects=OBJECTS, locations=LOCATIONS, depth=2)
SCHEMA_VERSION = "paper2.world.v2"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def answer_label(location: str) -> str:
    try:
        return next(label for label, value in CHOICES.items() if value == location)
    except StopIteration as exc:
        raise ValueError(f"Unknown location: {location}") from exc


def event_dict(event: Event) -> dict[str, Any]:
    return {"kind": event.kind, "object": event.object, "location": event.location,
            "agents": sorted(event.agents)}


def event_from_dict(value: Mapping[str, Any]) -> Event:
    event = Event(kind=value["kind"], object=value["object"], location=value["location"],
                  agents=frozenset(value["agents"]))
    event.validate(CONFIG)
    return event


@dataclass(frozen=True)
class StoryWorld:
    initial_locations: dict[str, str]
    events: tuple[Event, ...]
    critical_event_index: int
    focal_word: tuple[str, ...]
    focal_object: str
    distractor_object: str
    overwrite_location: str
    donor_location: str
    split: str = "development"

    def __post_init__(self) -> None:
        EpistemicState.initial(CONFIG, self.initial_locations)
        if not 0 <= self.critical_event_index < len(self.events):
            raise ValueError("Critical event index is outside the event sequence")
        if not 1 <= len(self.focal_word) <= 2 or set(self.focal_word) - set(AGENTS):
            raise ValueError("Focal belief word must contain one or two known characters")
        if self.focal_object == self.distractor_object:
            raise ValueError("The irrelevant object must be distinct")
        if self.overwrite_location not in LOCATIONS or self.donor_location not in LOCATIONS:
            raise ValueError("Unknown overwrite/donor location")
        if self.critical_event.object != self.focal_object:
            raise ValueError("The critical event must concern the focal object")
        if not set(self.focal_word).issubset(self.critical_event.agents):
            raise ValueError("The critical event must update the queried belief")
        for event in self.events:
            event.validate(CONFIG)

    @property
    def critical_event(self) -> Event:
        return self.events[self.critical_event_index]

    @property
    def agent(self) -> str:
        return self.focal_word[0]

    @property
    def other(self) -> str:
        return next(agent for agent in AGENTS if agent != self.agent)

    @property
    def world_id(self) -> str:
        return digest(self.semantic_payload())[:24]

    @property
    def family_id(self) -> str:
        """Collapse critical-value twins, split and unused donor metadata."""
        payload = self.semantic_payload()
        payload["events"][self.critical_event_index]["location"] = "<critical-value>"
        return digest(payload)[:24]

    def semantic_payload(self) -> dict[str, Any]:
        return {"initial_locations": dict(self.initial_locations),
                "events": [event_dict(event) for event in self.events],
                "critical_event_index": self.critical_event_index,
                "focal_word": list(self.focal_word), "focal_object": self.focal_object,
                "distractor_object": self.distractor_object,
                "overwrite_location": self.overwrite_location}

    def to_dict(self) -> dict[str, Any]:
        return {**self.semantic_payload(), "donor_location": self.donor_location, "split": self.split}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StoryWorld":
        return cls(initial_locations=dict(value["initial_locations"]),
                   events=tuple(event_from_dict(event) for event in value["events"]),
                   critical_event_index=int(value["critical_event_index"]),
                   focal_word=tuple(value["focal_word"]), focal_object=value["focal_object"],
                   distractor_object=value["distractor_object"],
                   overwrite_location=value["overwrite_location"], donor_location=value["donor_location"],
                   split=value.get("split", "development"))

    def state(self) -> EpistemicState:
        state = EpistemicState.initial(CONFIG, self.initial_locations)
        for event in self.events:
            state = state.apply(event)
        return state


def generate_world(rng: random.Random, *, split: str = "development", depth: int = 1,
                   event_kind: str | None = None, distractor_events: int = 2,
                   object_index: int | None = None, hidden_move: bool = False,
                   target_location: str | None = None) -> StoryWorld:
    if depth not in (1, 2):
        raise ValueError("Paper 2 supports belief depth one or two")
    if distractor_events < 0:
        raise ValueError("distractor_events must be nonnegative")
    initial = {obj: rng.choice(LOCATIONS) for obj in OBJECTS}
    obj = OBJECTS[object_index % len(OBJECTS)] if object_index is not None else rng.choice(OBJECTS)
    distractor = rng.choice([candidate for candidate in OBJECTS if candidate != obj])
    agent = rng.choice(AGENTS)
    other = next(candidate for candidate in AGENTS if candidate != agent)
    word = (agent,) if depth == 1 else (agent, other)
    values = [loc for loc in LOCATIONS if loc != initial[obj]]
    if target_location is not None:
        if target_location not in LOCATIONS:
            raise ValueError("Unknown target location")
        if initial[obj] == target_location:
            initial[obj] = rng.choice([loc for loc in LOCATIONS if loc != target_location])
        base = target_location
        donor, overwrite = rng.sample([loc for loc in LOCATIONS if loc not in (base, initial[obj])], 2)
    else:
        base, donor, overwrite = rng.sample(values, 3)
    critical = Event(kind=event_kind or rng.choice(("move", "tell")), object=obj,
                     location=base, agents=frozenset(word))
    background = []
    available = [candidate for candidate in OBJECTS if candidate not in (obj, distractor)]
    for _ in range(distractor_events):
        observers = frozenset(rng.sample(list(AGENTS), rng.randint(1, 2)))
        background.append(Event(kind=rng.choice(("move", "tell")), object=rng.choice(available),
                                location=rng.choice(LOCATIONS), agents=observers))
    index = distractor_events // 2
    events = background[:index] + [critical] + background[index:]
    if hidden_move:
        # An unobserved move changes the physical world without changing the focal word.
        events.append(Event(kind="move", object=obj, location=rng.choice([x for x in LOCATIONS if x != base]),
                            agents=frozenset({other})))
    return StoryWorld(initial, tuple(events), index, word, obj, distractor, overwrite, donor, split)


def _view_state(world: StoryWorld, view: str, state: EpistemicState | None = None) -> EpistemicState:
    if view not in VIEWS:
        raise ValueError(f"Unknown consequence view: {view}")
    state = world.state() if state is None else state
    if view in ("observed_overwrite", "unobserved_overwrite"):
        observers = frozenset(world.focal_word) if view == "observed_overwrite" else frozenset({world.other})
        return state.apply(Event("move", world.focal_object, world.overwrite_location, observers))
    if view == "report":
        value = state.get(world.focal_word, world.focal_object)
        return state.apply(Event("tell", world.focal_object, value, frozenset({world.other})))
    return state


def ordinary_target(world: StoryWorld, view: str, state: EpistemicState | None = None) -> str:
    state = _view_state(world, view, state)
    if view == "world":
        return state.get((), world.focal_object)
    if view in ("other_agent", "report"):
        return state.get((world.other,), world.focal_object)
    if view == "irrelevant_object":
        return state.get(world.focal_word, world.distractor_object)
    return state.get(world.focal_word, world.focal_object)


def _belief_question(word: tuple[str, ...], obj: str) -> str:
    if len(word) == 1:
        return f"Where does {word[0]} believe the {obj} is?"
    return f"Where does {word[0]} believe {word[1]} believes the {obj} is?"


def _story_and_query(world: StoryWorld, view: str) -> tuple[str, str, list[int], list[int]]:
    a, b, obj = world.agent, world.other, world.focal_object
    story = verbalize_initial(world.initial_locations)
    event_span = value_span = None
    for index, event in enumerate(world.events):
        text = verbalize_event(event, CONFIG)
        start = len(story) + 1
        story += " " + text
        if index == world.critical_event_index:
            event_span = [start, start + len(text)]
            location_start = start + text.index("the " + event.location) + 4
            value_span = [location_start, location_start + len(event.location)]
    question = _belief_question(world.focal_word, obj)
    if view == "world":
        question = f"Where is the {obj} actually located?"
    elif view == "other_agent":
        question = _belief_question((b,), obj)
    elif view in ("observed_overwrite", "unobserved_overwrite"):
        observers = frozenset(world.focal_word) if view == "observed_overwrite" else frozenset({b})
        event = Event("move", obj, world.overwrite_location, observers)
        story += "\nNext, " + verbalize_event(event, CONFIG) + f" This is the final move of the {obj}."
        question = "After both moves, " + question[0].lower() + question[1:]
    elif view == "search":
        if len(world.focal_word) == 1:
            story += (f"\nAfter all events described above, {a} begins a new search for the {obj}. "
                      f"{a} first checks the place where {a} currently believes the {obj} is.")
            question = f"Where does {a} check first in this new search?"
        else:
            story += (f"\nAfter all events above, {b} begins a new search for the {obj}. "
                      f"{a} expects {b} to first check the location where {a} believes {b} believes the {obj} is.")
            question = f"Where does {a} expect {b} to check first?"
    elif view == "irrelevant_object":
        question = _belief_question(world.focal_word, world.distractor_object)
    elif view == "report":
        premise = (f"{a} currently believes the {obj} is" if len(world.focal_word) == 1
                   else f"{a} currently believes {b} believes the {obj} is")
        story += (f"\nAfter the events above, {a} tells {b} the location where {premise}. "
                  f"{a} accurately communicates this current belief. "
                  f"{b} accepts the report and updates {b}'s belief accordingly. "
                  "No object moves during this conversation.")
        question = f"After this conversation, where does {b} now believe the {obj} is?"
    if view not in VIEWS:
        raise ValueError(f"Unknown consequence view: {view}")
    assert event_span is not None and value_span is not None
    return story, question, event_span, value_span


def cue_order(world: StoryWorld, cue_condition: str, cue_location: str | None = None) -> tuple[str, ...]:
    """Assign a world-level cue against the rewarded DIRECT-belief target.

    Never reassign the cue after seeing a consequence question. In particular,
    an agreeing cue can disagree with an overwrite/world/irrelevant target.
    """
    if cue_condition not in ("none", "clean", "agree", "independent", "conflict", "shortcut"):
        raise ValueError(f"Unknown cue condition: {cue_condition}")
    target = ordinary_target(world, "direct")
    rng = random.Random(int(digest({"world": world.world_id, "purpose": "inventory"}), 16))
    if cue_location is None:
        if cue_condition == "agree":
            cue_location = target
        elif cue_condition == "conflict":
            cue_location = rng.choice([loc for loc in LOCATIONS if loc != target])
        else:
            cue_location = rng.choice(LOCATIONS)
    if cue_location not in LOCATIONS:
        raise ValueError("Cue location must be a known location")
    if cue_condition == "agree" and cue_location != target:
        raise ValueError("Agree condition must agree")
    if cue_condition == "conflict" and cue_location == target:
        raise ValueError("Conflict condition must conflict")
    rest = [loc for loc in LOCATIONS if loc != cue_location]
    rng.shuffle(rest)
    return tuple(rest + [cue_location])


def render_record(world: StoryWorld, view: str, cue_condition: str = "none", *,
                  cue_location: str | None = None, dataset: str = "clean", include_world: bool = True) -> dict[str, Any]:
    story, question, event_span, value_span = _story_and_query(world, view)
    base_story = _story_and_query(world, "direct")[0]
    target = ordinary_target(world, view)
    direct_target = ordinary_target(world, "direct")
    cue = None
    if cue_condition != "none":
        order = cue_order(world, cue_condition, cue_location)
        # Permuting whole, equally formatted sentences preserves character count,
        # word count and every entity frequency. These are labels, never events.
        appendix = "\nUnrelated inventory labels; these labels convey no object locations. "
        appendix += " ".join(f"Unused label {location}." for location in order)
        story += appendix
        base_story += appendix
        cue = order[-1]
    target_scope = "belief_dependent" if view in ("direct", "search", "unobserved_overwrite", "report") else "preservation"
    query_word = () if view == "world" else ((world.other,) if view in ("other_agent", "report") else world.focal_word)
    query_object = world.distractor_object if view == "irrelevant_object" else world.focal_object
    row = {"schema_version": SCHEMA_VERSION, "id": digest([world.world_id, view, cue_condition, cue])[:24],
           "world_id": world.world_id, "story_id": world.world_id, "family_id": world.family_id,
           "split": world.split, "dataset": dataset, "cue_condition": cue_condition,
           "cue_location": cue, "cue_agrees": None if cue is None else cue == direct_target,
           "cue_agrees_direct": None if cue is None else cue == direct_target,
           "cue_agrees_evaluated": None if cue is None else cue == target,
           "view": view, "story": story, "base_story": base_story, "question": question,
           "query": {"text": question, "kind": "direct" if view == "direct" else "consequence",
                     "word": list(query_word), "object": query_object},
           "choices": dict(CHOICES), "answer": answer_label(target), "target_location": target,
           "target_scope": target_scope, "belief_depth": len(world.focal_word),
           "critical_event_span": event_span, "critical_value_span": value_span,
           "critical_event_kind": world.critical_event.kind,
           "rendering_extension": "nested-consequence-projections-v1" if len(world.focal_word) == 2 else None}
    if include_world:
        row["world"] = world.to_dict()
    row["rendering_id"] = row["id"]
    row["cue_answer"] = None if cue is None else answer_label(cue)
    return row


def rewrite_world(world: StoryWorld, value: str) -> StoryWorld:
    if value not in LOCATIONS:
        raise ValueError("Unknown rewrite location")
    events = list(world.events)
    events[world.critical_event_index] = replace(world.critical_event, location=value)
    return replace(world, events=tuple(events))


def rewrite_forecast(world: StoryWorld, view: str, value: str, account: str = "event_rewrite", *,
                     objective: str = "f_star") -> str:
    """Location target for a specified swap account; never an ordinary target.

    Event rewriting replays the symbolic generator. Other accounts reproduce the
    six Paper 1 causal-account roles with the queried word replacing first-order
    belief for depth-two records. The optional wrong-map objective uses Paper 1's
    fixed transposition. The proposal's natural-donor swaps use f_star throughout.
    """
    if account not in ACCOUNTS or view not in VIEWS:
        raise ValueError("Unknown account or consequence view")
    if value not in LOCATIONS:
        raise ValueError("Unknown rewrite location")
    if objective not in ("f_star", "g_star"):
        raise ValueError("Unknown Paper 1 forecast objective")
    permutation = dict(zip(LOCATIONS, ("basket", "box", "drawer", "shelf", "closet", "cabinet")))
    if account == "global_output_permutation":
        natural = ordinary_target(rewrite_world(world, value), view)
        return natural if objective == "f_star" else permutation[natural]
    value = value if objective == "f_star" else permutation[value]
    if account == "event_rewrite":
        return ordinary_target(rewrite_world(world, value), view)
    if account in ("persistent_direct_override", "cleared_direct_override"):
        if view == "direct" or (account == "persistent_direct_override" and view in ("observed_overwrite", "unobserved_overwrite")):
            return value
        return ordinary_target(world, view)
    state = world.state()
    values = dict(state.values)
    if account == "belief_edit":
        values[(world.focal_word, world.focal_object)] = value
    elif account == "broad_state_write":
        for word in CONFIG.words():
            values[(word, world.focal_object)] = value
    return ordinary_target(world, view, EpistemicState(CONFIG, values))


def matched_pair(world: StoryWorld, event_kind: str | None = None) -> dict[str, Any]:
    if event_kind is not None:
        if event_kind not in ("move", "tell"):
            raise ValueError("Matched pairs must use observation or communication")
        events = list(world.events)
        events[world.critical_event_index] = replace(world.critical_event, kind=event_kind)
        world = replace(world, events=tuple(events))
    donor = rewrite_world(world, world.donor_location)
    if donor.critical_event.location == world.critical_event.location:
        raise ValueError("A natural donor must differ at the critical event")
    forecasts = {view: {account: answer_label(rewrite_forecast(world, view, world.donor_location, account))
                        for account in ACCOUNTS} for view in VIEWS}
    recipient_rows = [render_record(world, view) for view in VIEWS]
    donor_rows = [render_record(donor, view) for view in VIEWS]
    return {"schema_version": SCHEMA_VERSION, "pair_id": digest([world.world_id, donor.world_id])[:24],
            "world_id": world.world_id, "family_id": world.family_id, "split": world.split,
            "event_kind": world.critical_event.kind, "changed_event_index": world.critical_event_index,
            "recipient_world": world.to_dict(), "donor_world": donor.to_dict(),
            "recipient": recipient_rows, "donor": donor_rows, "forecasts": forecasts,
            "event_rewrite_change_mask": {view: forecasts[view]["event_rewrite"] != recipient_rows[i]["answer"]
                                           for i, view in enumerate(VIEWS)},
            "forecast_scope": "Only Test 3 specified swaps; Tests 1/2 use ordinary targets"}


def entity_frequencies(text: str) -> dict[str, int]:
    words = Counter(re.findall(r"\b\w+\b", text))
    return {entity: words[entity] for entity in AGENTS + OBJECTS + LOCATIONS}


def validate_cue_renderings(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("At least one rendering is required")
    if len({(row["world_id"], row["view"]) for row in rows}) != 1:
        raise ValueError("Cue renderings must share the underlying world and question")
    lengths = {(len(row["story"]), len(row["story"].split())) for row in rows}
    frequencies = {json.dumps(entity_frequencies(row["story"]), sort_keys=True) for row in rows}
    if len(lengths) != 1 or len(frequencies) != 1:
        raise ValueError("Cue renderings differ in length or entity frequency")
    for row in rows:
        mentions = re.findall(r"\b(?:" + "|".join(LOCATIONS) + r")\b", row["story"])
        if not mentions or mentions[-1] != row["cue_location"]:
            raise ValueError("Cue is not the last-mentioned story location")
        if row["cue_agrees"] != row["cue_agrees_direct"]:
            raise ValueError("Cue agreement alias must use the rewarded direct target")
        if row["cue_agrees_evaluated"] != (row["cue_location"] == row["target_location"]):
            raise ValueError("Evaluated-answer cue agreement metadata is incorrect")
        if row["cue_condition"] == "agree" and not row["cue_agrees_direct"]:
            raise ValueError("Agree rendering is incorrect")
        if row["cue_condition"] == "conflict" and row["cue_agrees_direct"]:
            raise ValueError("Conflict rendering is incorrect")
    return {"matched": True, "character_length": len(rows[0]["story"]),
            "word_length": len(rows[0]["story"].split()), "entity_frequencies": entity_frequencies(rows[0]["story"])}


def build_oracle_trace(row: Mapping[str, Any]) -> str:
    """Extra symbolic supervision available only to reasoning SFT.

    This stated source is deliberately not presented as a base-model-generated
    trace. It exposes the task transition rule and must be reported as additional
    information when comparing reasoning SFT with outcome-only RL.
    """
    if row["view"] != "direct":
        raise ValueError("Training trace supervision is limited to the direct view")
    world = StoryWorld.from_dict(row["world"])
    obj = world.focal_object
    belief = world.initial_locations[obj]
    parts = [f"Initially everyone sees the {obj} in the {belief}, so the queried belief starts at {belief}."]
    for event in world.events:
        if event.object != obj:
            continue
        observers = ", ".join(sorted(event.agents)) or "no characters"
        if set(world.focal_word).issubset(event.agents):
            belief = event.location
            parts.append(f"The {event.kind} event at {event.location} reaches {observers}; all characters in the queried belief observe or receive it, so that belief becomes {belief}.")
        else:
            parts.append(f"The {event.kind} event at {event.location} reaches only {observers}; it does not reach every character in the queried belief, so that belief remains {belief}.")
    parts.append(f"The queried belief is therefore {belief}, corresponding to choice {answer_label(belief)}.")
    return " ".join(parts)
