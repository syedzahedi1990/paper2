






from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Literal


Word = tuple[str, ...]
Coordinate = tuple[Word, str]
EventKind = Literal["move", "tell"]
QueryKind = Literal["direct", "action", "false", "counterfactual"]


def _require_unique(name: str, values: tuple[str, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique: {values}")
    if any(not value for value in values):
        raise ValueError(f"{name} must not contain empty strings")


def _format_list(items: Iterable[str]) -> str:
    values = list(items)
    if not values:
        return "no one"
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _negative_info_sentence(agents: list[str], message: str) -> str:
    if not agents:
        return ""
    subject = _format_list(agents)
    verb = "does" if len(agents) == 1 else "do"
    return f" {subject} {verb} not {message}."


@dataclass(frozen=True)
class TaskConfig:
    agents: tuple[str, ...]
    objects: tuple[str, ...]
    locations: tuple[str, ...]
    depth: int

    def __post_init__(self) -> None:
        _require_unique("agents", self.agents)
        _require_unique("objects", self.objects)
        _require_unique("locations", self.locations)
        if self.depth < 0:
            raise ValueError("depth must be nonnegative")
        if not self.agents:
            raise ValueError("at least one agent is required")
        if not self.objects:
            raise ValueError("at least one object is required")
        if not self.locations:
            raise ValueError("at least one location is required")

    def words(self) -> tuple[Word, ...]:
        words: list[Word] = [()]
        for size in range(1, self.depth + 1):
            words.extend(itertools.product(self.agents, repeat=size))
        return tuple(words)

    def coordinates(self) -> tuple[Coordinate, ...]:
        return tuple((word, obj) for word in self.words() for obj in self.objects)

    def direct_queries(self) -> tuple["Query", ...]:
        return tuple(
            Query(kind="direct", word=word, object=obj)
            for word in self.words()
            for obj in self.objects
        )


@dataclass(frozen=True)
class Event:
    kind: EventKind
    object: str
    location: str
    agents: frozenset[str]

    def validate(self, config: TaskConfig) -> None:
        if self.kind not in ("move", "tell"):
            raise ValueError(f"unknown event kind: {self.kind}")
        if self.object not in config.objects:
            raise ValueError(f"unknown object: {self.object}")
        if self.location not in config.locations:
            raise ValueError(f"unknown location: {self.location}")
        unknown = self.agents.difference(config.agents)
        if unknown:
            raise ValueError(f"unknown event agents: {sorted(unknown)}")

    def to_record(self, config: TaskConfig | None = None) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "object": self.object,
            "location": self.location,
            "agents": sorted(self.agents),
            "text": verbalize_event(self, config),
        }


@dataclass(frozen=True)
class Query:
    kind: QueryKind
    object: str
    word: Word = ()
    event: Event | None = None

    def validate(self, config: TaskConfig) -> None:
        if self.kind not in ("direct", "action", "false", "counterfactual"):
            raise ValueError(f"unknown query kind: {self.kind}")
        if self.object not in config.objects:
            raise ValueError(f"unknown query object: {self.object}")
        if len(self.word) > config.depth:
            raise ValueError(f"word exceeds configured depth: {self.word}")
        unknown = set(self.word).difference(config.agents)
        if unknown:
            raise ValueError(f"unknown query agents: {sorted(unknown)}")
        if self.kind == "action" and len(self.word) != 1:
            raise ValueError("action queries require a one-agent word")
        if self.kind == "counterfactual":
            if self.event is None:
                raise ValueError("counterfactual queries require an event")
            self.event.validate(config)
        elif self.event is not None:
            raise ValueError("only counterfactual queries may include an event")

    def to_record(self, config: TaskConfig) -> dict[str, Any]:
        self.validate(config)
        record: dict[str, Any] = {
            "kind": self.kind,
            "object": self.object,
            "word": list(self.word),
            "text": verbalize_query(self),
        }
        if self.event is not None:
            record["event"] = self.event.to_record()
        return record


@dataclass(frozen=True)
class EpistemicState:
    config: TaskConfig
    values: dict[Coordinate, str]

    @classmethod
    def initial(
        cls,
        config: TaskConfig,
        initial_locations: dict[str, str] | None = None,
        rng: random.Random | None = None,
    ) -> "EpistemicState":
        if initial_locations is None:
            rng = rng or random.Random()
            initial_locations = {
                obj: rng.choice(config.locations) for obj in config.objects
            }
        missing = set(config.objects).difference(initial_locations)
        if missing:
            raise ValueError(f"missing initial locations for: {sorted(missing)}")
        unknown = set(initial_locations).difference(config.objects)
        if unknown:
            raise ValueError(f"unknown initial-location objects: {sorted(unknown)}")
        for obj, loc in initial_locations.items():
            if loc not in config.locations:
                raise ValueError(f"unknown initial location for {obj}: {loc}")
        values = {
            (word, obj): initial_locations[obj]
            for word in config.words()
            for obj in config.objects
        }
        return cls(config=config, values=values)

    def __post_init__(self) -> None:
        expected = set(self.config.coordinates())
        got = set(self.values)
        if expected != got:
            missing = expected.difference(got)
            extra = got.difference(expected)
            raise ValueError(f"state coordinate mismatch: missing={missing}, extra={extra}")
        bad_locations = {
            loc for loc in self.values.values() if loc not in self.config.locations
        }
        if bad_locations:
            raise ValueError(f"unknown state locations: {sorted(bad_locations)}")

    def get(self, word: Word, obj: str) -> str:
        return self.values[(tuple(word), obj)]

    def true_locations(self) -> dict[str, str]:
        return {obj: self.get((), obj) for obj in self.config.objects}

    def apply(self, event: Event) -> "EpistemicState":
        event.validate(self.config)
        values = dict(self.values)
        if event.kind == "move":
            values[((), event.object)] = event.location
        for word in self.config.words():
            if not word:
                continue
            if set(word).issubset(event.agents):
                values[(word, event.object)] = event.location
        return EpistemicState(config=self.config, values=values)

    def evaluate(self, query: Query) -> str | bool:
        query.validate(self.config)
        if query.kind == "direct":
            return self.get(query.word, query.object)
        if query.kind == "action":
            return self.get(query.word, query.object)
        if query.kind == "false":
            return self.get(query.word, query.object) != self.get((), query.object)
        if query.kind == "counterfactual":
            assert query.event is not None
            return self.apply(query.event).get(query.word, query.object)
        raise AssertionError(f"unreachable query kind: {query.kind}")

    def to_record(self) -> dict[str, str]:
        return {
            coordinate_key(word, obj): loc
            for (word, obj), loc in sorted(
                self.values.items(), key=lambda item: coordinate_key(*item[0])
            )
        }

    def differing_coordinates(self, other: "EpistemicState") -> tuple[Coordinate, ...]:
        if self.config != other.config:
            raise ValueError("cannot compare states with different configs")
        return tuple(
            coord for coord in self.config.coordinates() if self.values[coord] != other.values[coord]
        )


@dataclass(frozen=True)
class Trace:
    config: TaskConfig
    initial_locations: dict[str, str]
    events: tuple[Event, ...]

    def initial_state(self) -> EpistemicState:
        return EpistemicState.initial(self.config, dict(self.initial_locations))

    def final_state(self) -> EpistemicState:
        state = self.initial_state()
        for event in self.events:
            state = state.apply(event)
        return state

    def story(self) -> str:
        return verbalize_trace(self)

    def to_record(self) -> dict[str, Any]:
        return {
            "initial_locations": dict(self.initial_locations),
            "events": [event.to_record(self.config) for event in self.events],
            "story": self.story(),
            "final_state": self.final_state().to_record(),
        }


@dataclass(frozen=True)
class BlockPair:
    plus: Trace
    minus: Trace
    block: tuple[Coordinate, ...]

    def diff_coordinates(self) -> tuple[Coordinate, ...]:
        return self.plus.final_state().differing_coordinates(self.minus.final_state())

    def to_record(self) -> dict[str, Any]:
        return {
            "plus": self.plus.to_record(),
            "minus": self.minus.to_record(),
            "block": [coordinate_key(word, obj) for word, obj in self.block],
            "diff_coordinates": [
                coordinate_key(word, obj) for word, obj in self.diff_coordinates()
            ],
        }


def coordinate_key(word: Word, obj: str) -> str:
    prefix = "epsilon" if not word else ".".join(word)
    return f"{prefix}|{obj}"


def word_phrase(word: Word) -> str:
    if not word:
        return "really"
    if len(word) == 1:
        return f"{word[0]} thinks"
    return " thinks ".join(word) + " thinks"


def question_belief_phrase(word: Word) -> str:
    if not word:
        return "really"
    if len(word) == 1:
        return f"{word[0]} think"
    inner = " ".join(f"{agent} thinks" for agent in word[1:])
    return f"{word[0]} think {inner}"


def verbalize_initial(initial_locations: dict[str, str]) -> str:
    sentences = [
        f"Everyone initially sees that the {obj} is in the {loc}."
        for obj, loc in sorted(initial_locations.items())
    ]
    return " ".join(sentences)


def verbalize_event(event: Event, config: TaskConfig | None) -> str:
    event_agents = sorted(event.agents)
    agents = _format_list(event_agents)
    if config is None:
        absent: list[str] = []
    else:
        absent = [agent for agent in config.agents if agent not in event.agents]
    if event.kind == "move":
        unseen = _negative_info_sentence(absent, "see this happen")
        if event.agents:
            verb = "watches" if len(event_agents) == 1 else "watch"
            return (
                f"{agents} {verb} as the {event.object} is moved to the "
                f"{event.location}.{unseen}"
            )
        return f"No one sees the {event.object} being moved to the {event.location}."
    unseen = _negative_info_sentence(absent, "receive this information")
    if event.agents:
        verb = "is" if len(event_agents) == 1 else "are"
        return (
            f"{agents} {verb} told that the {event.object} is in the "
            f"{event.location}.{unseen}"
        )
    return f"No one is told that the {event.object} is in the {event.location}."


def verbalize_trace(trace: Trace) -> str:
    sentences = [verbalize_initial(trace.initial_locations)]
    sentences.extend(verbalize_event(event, trace.config) for event in trace.events)
    return " ".join(sentences)


def verbalize_query(query: Query) -> str:
    if query.kind == "direct":
        if not query.word:
            return f"Where is the {query.object} really?"
        return f"Where does {question_belief_phrase(query.word)} the {query.object} is?"
    if query.kind == "action":
        return f"Where will {query.word[0]} look for the {query.object}?"
    if query.kind == "false":
        if not query.word:
            return f"Is the true location of the {query.object} false?"
        return (
            f"Does {question_belief_phrase(query.word)} the {query.object} "
            "is in the wrong place?"
        )
    if query.kind == "counterfactual":
        assert query.event is not None
        event_text = verbalize_event(query.event, None).rstrip(".")
        if not query.word:
            return f"If {event_text}, where would the {query.object} really be?"
        return (
            f"If {event_text}, where would {question_belief_phrase(query.word)} "
            f"the {query.object} is?"
        )
    raise AssertionError(f"unreachable query kind: {query.kind}")


def changed_block(config: TaskConfig, event: Event) -> tuple[Coordinate, ...]:
    event.validate(config)
    coords: list[Coordinate] = []
    if event.kind == "move":
        coords.append(((), event.object))
    coords.extend(
        (word, event.object)
        for word in config.words()
        if word and set(word).issubset(event.agents)
    )
    return tuple(coords)


def sample_event(
    config: TaskConfig,
    rng: random.Random,
    kind: EventKind | None = None,
    obj: str | None = None,
) -> Event:
    if kind is None:
        kind = rng.choice(("move", "tell"))
    if obj is None:
        obj = rng.choice(config.objects)
    location = rng.choice(config.locations)
    agents = frozenset(
        agent for agent in config.agents if rng.random() < 0.5
    )
    if kind == "tell" and not agents:
        agents = frozenset({rng.choice(config.agents)})
    return Event(kind=kind, object=obj, location=location, agents=agents)


def sample_trace(config: TaskConfig, length: int, rng: random.Random) -> Trace:
    initial_locations = {
        obj: rng.choice(config.locations) for obj in config.objects
    }
    events = tuple(sample_event(config, rng) for _ in range(length))
    return Trace(config=config, initial_locations=initial_locations, events=events)


def sample_query(
    config: TaskConfig,
    rng: random.Random,
    kinds: tuple[QueryKind, ...] = ("direct", "action", "false"),
) -> Query:
    kind = rng.choice(kinds)
    obj = rng.choice(config.objects)
    if kind == "direct":
        return Query(kind=kind, object=obj, word=rng.choice(config.words()))
    if kind == "action":
        return Query(kind=kind, object=obj, word=(rng.choice(config.agents),))
    if kind == "false":
        nonempty_words = [word for word in config.words() if word]
        return Query(kind=kind, object=obj, word=rng.choice(nonempty_words))
    if kind == "counterfactual":
        return Query(
            kind=kind,
            object=obj,
            word=rng.choice(config.words()),
            event=sample_event(config, rng, obj=obj),
        )
    raise ValueError(f"unknown query kind: {kind}")


def example_record(index: int, trace: Trace, query: Query) -> dict[str, Any]:
    state = trace.final_state()
    answer = state.evaluate(query)
    answer_type = "boolean" if isinstance(answer, bool) else "location"
    choices = ["yes", "no"] if answer_type == "boolean" else list(trace.config.locations)
    answer_text = ("yes" if answer else "no") if isinstance(answer, bool) else answer
    return {
        "id": index,
        **trace.to_record(),
        "query": query.to_record(trace.config),
        "answer": answer,
        "answer_text": answer_text,
        "answer_type": answer_type,
        "choices": choices,
    }


def generate_examples(
    config: TaskConfig,
    count: int,
    trace_length: int,
    seed: int,
    query_kinds: tuple[QueryKind, ...] = ("direct", "action", "false"),
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    for index in range(count):
        trace = sample_trace(config, trace_length, rng)
        query = sample_query(config, rng, query_kinds)
        records.append(example_record(index, trace, query))
    return records


def make_block_pair(
    config: TaskConfig,
    initial_locations: dict[str, str],
    prefix_events: tuple[Event, ...],
    kind: EventKind,
    obj: str,
    agents: frozenset[str],
    minus_location: str,
    plus_location: str,
    suffix_events: tuple[Event, ...] = (),
) -> BlockPair:
    if minus_location == plus_location:
        raise ValueError("minus_location and plus_location must differ")
    for event in suffix_events:
        if event.object == obj:
            raise ValueError("suffix events must not touch the paired object")
    minus_event = Event(kind=kind, object=obj, location=minus_location, agents=agents)
    plus_event = Event(kind=kind, object=obj, location=plus_location, agents=agents)
    minus_event.validate(config)
    plus_event.validate(config)
    minus = Trace(
        config=config,
        initial_locations=dict(initial_locations),
        events=tuple(prefix_events) + (minus_event,) + tuple(suffix_events),
    )
    plus = Trace(
        config=config,
        initial_locations=dict(initial_locations),
        events=tuple(prefix_events) + (plus_event,) + tuple(suffix_events),
    )
    block = changed_block(config, plus_event)
    return BlockPair(plus=plus, minus=minus, block=block)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--trace-length", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--agents", nargs="+", default=["Alice", "Bob"])
    parser.add_argument("--objects", nargs="+", default=["key"])
    parser.add_argument("--locations", nargs="+", default=["box", "basket", "shelf"])
    parser.add_argument(
        "--query-kinds",
        nargs="+",
        choices=["direct", "action", "false", "counterfactual"],
        default=["direct", "action", "false"],
    )
    parser.add_argument("--output", default="-", help="JSONL path, or '-' for stdout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    config = TaskConfig(
        agents=tuple(args.agents),
        objects=tuple(args.objects),
        locations=tuple(args.locations),
        depth=args.depth,
    )
    records = generate_examples(
        config=config,
        count=args.examples,
        trace_length=args.trace_length,
        seed=args.seed,
        query_kinds=tuple(args.query_kinds),
    )
    lines = [json.dumps(record, sort_keys=True) for record in records]
    if args.output == "-":
        for line in lines:
            print(line)
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
