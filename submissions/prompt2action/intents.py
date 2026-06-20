from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


SUPPORTED_INTENTS = (
    "wave",
    "walk_forward",
    "step_back",
    "turn_left",
    "turn_right",
    "bow",
    "stop",
    "idle",
)

INTENT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "wave": ("wave", "say hello", "hello", "greet", "raise your hand"),
    "walk_forward": ("walk", "walk forward", "move forward", "go forward", "advance", "march"),
    "step_back": ("step back", "move back", "go back", "back up", "reverse", "walk back", "walk backward"),
    "turn_left": ("turn left", "rotate left", "face left", "spin left"),
    "turn_right": ("turn right", "rotate right", "face right", "spin right"),
    "bow": ("bow", "take a bow", "lean forward"),
    "stop": ("stop", "halt", "freeze", "cancel"),
    "idle": ("idle", "stand still", "neutral", "reset pose", "relax"),
}

HELP_TEXT = (
    "Supported actions: wave, walk forward, step back, turn left, turn right, "
    "bow, stop, idle. You can chain them, for example: 'turn left and walk forward'."
)

DEFAULT_REPEAT = 1

_NUMBER_WORDS = {
    "one": 1,
    "once": 1,
    "two": 2,
    "twice": 2,
    "three": 3,
    "thrice": 3,
    "four": 4,
}

_SEGMENT_SPLIT_RE = re.compile(r"\s*(?:,|;|\band then\b|\bthen\b|\band\b)\s*")


@dataclass(slots=True)
class ParsedAction:
    intent: str
    duration_s: float | None = None
    repeat: int = DEFAULT_REPEAT
    turn_degrees: float | None = None
    walk_distance_m: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ParsedCommand:
    actions: list[ParsedAction]
    raw_text: str
    confidence: float
    source: str
    message: str | None = None
    diagnostics: str | None = None

    @property
    def intent(self) -> str | None:
        return self.actions[0].intent if self.actions else None

    def to_dict(self) -> dict[str, object]:
        return {
            "actions": [action.to_dict() for action in self.actions],
            "raw_text": self.raw_text,
            "confidence": self.confidence,
            "source": self.source,
            "message": self.message,
            "diagnostics": self.diagnostics,
        }


class OllamaClient:
    def __init__(self, model_name: str, prompt_path: Path) -> None:
        self.model_name = model_name
        self.prompt_path = prompt_path

    def is_available(self) -> bool:
        return shutil.which("ollama") is not None

    def translate(self, command: str) -> dict[str, object]:
        prompt = self.prompt_path.read_text(encoding="utf-8").format(
            supported_intents=", ".join(SUPPORTED_INTENTS),
            command=command,
        )
        completed = subprocess.run(
            ["ollama", "run", self.model_name, prompt],
            check=True,
            capture_output=True,
            text=True,
        )
        response = completed.stdout.strip()
        payload_text = _extract_json_object(response)
        if payload_text is None:
            raise ValueError("Ollama response did not contain JSON")
        try:
            return json.loads(payload_text)
        except json.JSONDecodeError:
            salvaged = _salvage_payload(payload_text)
            if salvaged is None:
                raise
            return salvaged


class CommandTranslator:
    def __init__(
        self,
        *,
        ollama_model: str = "llama3.2:3b",
        use_llm: bool = True,
        prompt_path: Path | None = None,
        ollama_client: OllamaClient | None = None,
    ) -> None:
        self.use_llm = use_llm
        self.prompt_path = prompt_path or Path(__file__).with_name("ollama_intent_prompt.txt")
        self.ollama_client = ollama_client or OllamaClient(ollama_model, self.prompt_path)

    def parse(self, raw_text: str) -> ParsedCommand:
        text = raw_text.strip()
        if not text:
            return ParsedCommand(
                actions=[],
                raw_text=raw_text,
                confidence=0.0,
                source="fallback",
                message=HELP_TEXT,
            )

        fallback = self._fallback_parse(text)
        if self.use_llm and self.ollama_client.is_available():
            try:
                llm_result = self._coerce_payload(self.ollama_client.translate(text), text, "ollama")
                if llm_result.actions:
                    if (
                        not fallback.actions
                        or _action_intents(fallback.actions) == _action_intents(llm_result.actions)
                        or _action_intent_prefix(fallback.actions, llm_result.actions)
                    ):
                        return llm_result
                    fallback.diagnostics = (
                        f"Ollama disagreement: llm={_actions_signature(llm_result.actions)}, "
                        f"fallback={_actions_signature(fallback.actions)}. Using fallback."
                    )
                    return fallback
            except Exception as exc:
                if not fallback.actions:
                    fallback.diagnostics = f"Ollama failed: {type(exc).__name__}: {exc}"
                return fallback

        if self.use_llm and not self.ollama_client.is_available():
            fallback.diagnostics = "Ollama CLI not found on PATH"
        return fallback

    def _coerce_payload(self, payload: dict[str, object], raw_text: str, source: str) -> ParsedCommand:
        actions_payload = payload.get("actions")
        if not isinstance(actions_payload, list):
            actions_payload = []
        actions: list[ParsedAction] = []
        for item in actions_payload:
            if not isinstance(item, dict):
                continue
            intent = str(item.get("intent", "")).strip()
            if intent not in SUPPORTED_INTENTS:
                continue
            actions.append(
                ParsedAction(
                    intent=intent,
                    duration_s=_coerce_duration(item.get("duration_s")),
                    repeat=_coerce_repeat(item.get("repeat")),
                    turn_degrees=_coerce_turn_degrees(item.get("turn_degrees"), intent),
                    walk_distance_m=_coerce_walk_distance(item.get("walk_distance_m"), intent),
                )
            )
        if source == "ollama":
            actions = _enrich_ollama_actions_from_text(actions, raw_text)
            actions = _normalize_ollama_walk_actions(actions, raw_text)
        actions = _dedupe_actions(actions)

        return ParsedCommand(
            actions=actions,
            raw_text=raw_text,
            confidence=_coerce_confidence(payload.get("confidence")),
            source=source,
            message=None if actions else HELP_TEXT,
        )

    def _fallback_parse(self, raw_text: str) -> ParsedCommand:
        text = _normalize_text(raw_text)
        actions: list[ParsedAction] = []
        for segment in _split_segments(text):
            actions.extend(_parse_segment_actions(segment))
        deduped = _merge_consecutive_duplicates(actions)
        return ParsedCommand(
            actions=deduped,
            raw_text=raw_text,
            confidence=0.78 if deduped else 0.0,
            source="fallback",
            message=None if deduped else HELP_TEXT,
        )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _split_segments(text: str) -> list[str]:
    return [segment for segment in _SEGMENT_SPLIT_RE.split(text) if segment]


def _parse_segment_actions(segment: str) -> list[ParsedAction]:
    duration_s = _extract_duration(segment)
    repeat = _extract_repeat(segment)
    turn_degrees = _extract_turn_degrees(segment)
    walk_distance_m = _extract_walk_distance(segment)
    found: list[tuple[int, str]] = []
    for intent, phrases in INTENT_SYNONYMS.items():
        for phrase in (intent, *phrases):
            index = segment.find(phrase)
            if index >= 0:
                found.append((index, intent))
    turn_match = re.search(r"\b(?:turn|rotate|spin|face)\b(?:(?!\b(?:turn|rotate|spin|face)\b).)*\b(left|right)\b", segment)
    if turn_match:
        found.append((turn_match.start(), f"turn_{turn_match.group(1)}"))
    if not any(intent in {"walk_forward", "step_back"} for _, intent in found):
        walk_direction_match = re.search(r"\b(forward|back|backward)\b", segment)
        if walk_direction_match:
            inferred_intent = "step_back" if walk_direction_match.group(1).startswith("back") else "walk_forward"
            found.append((walk_direction_match.start(), inferred_intent))
    if re.search(r"\b(?:walk|move|go)\s+back(?:ward)?\b", segment):
        found = [item for item in found if item[1] != "walk_forward"]
    found.sort(key=lambda item: item[0])

    actions: list[ParsedAction] = []
    seen: set[str] = set()
    for _, intent in found:
        if intent in seen:
            continue
        actions.append(
            ParsedAction(
                intent=intent,
                duration_s=duration_s,
                repeat=repeat,
                turn_degrees=turn_degrees if intent in {"turn_left", "turn_right"} else None,
                walk_distance_m=_resolve_walk_distance_for_intent(intent, walk_distance_m),
            )
        )
        seen.add(intent)
    return actions


def _merge_consecutive_duplicates(actions: list[ParsedAction]) -> list[ParsedAction]:
    if not actions:
        return []
    merged: list[ParsedAction] = [actions[0]]
    for action in actions[1:]:
        prev = merged[-1]
        if (
            prev.intent == action.intent
            and prev.duration_s == action.duration_s
            and prev.repeat == action.repeat
            and prev.turn_degrees == action.turn_degrees
            and prev.walk_distance_m == action.walk_distance_m
        ):
            continue
        merged.append(action)
    return merged


def _dedupe_actions(actions: list[ParsedAction]) -> list[ParsedAction]:
    deduped = _merge_consecutive_duplicates(actions)
    unique: list[ParsedAction] = []
    seen: set[tuple[str, float | None, int, float | None, float | None]] = set()
    for action in deduped:
        key = (action.intent, action.duration_s, action.repeat, action.turn_degrees, action.walk_distance_m)
        if key in seen:
            continue
        unique.append(action)
        seen.add(key)
    return unique


def _extract_duration(text: str) -> float | None:
    match = re.search(r"\bfor\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b", text)
    if not match:
        match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b", text)
    if not match:
        return None
    return round(float(match.group(1)), 2)


def _extract_repeat(text: str) -> int:
    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            return value
    match = re.search(r"\b(\d+)\s*(?:times|x|reps?)\b", text)
    if match:
        return max(1, min(5, int(match.group(1))))
    return DEFAULT_REPEAT


def _extract_turn_degrees(text: str) -> float | None:
    degree_patterns = (
        r"\bturn\s+(?:left|right)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*(?:degrees?|deg)\b",
        r"\bturn\s+(\d+(?:\.\d+)?)\s*(?:degrees?|deg)\s+(?:to\s+the\s+)?(left|right)\b",
        r"\b(?:rotate|spin)\s+(?:left|right)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*(?:degrees?|deg)\b",
    )
    for pattern in degree_patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        for group in match.groups():
            if group and re.fullmatch(r"\d+(?:\.\d+)?", group):
                return _clamp_turn_degrees(float(group))

    qualitative_turns = (
        (r"\b(?:a\s+tiny\s+bit|tiny|just\s+a\s+tiny\s+bit)\b", 5.0),
        (r"\b(?:a\s+little|little|slightly)\b", 15.0),
        (r"\b(?:a\s+bit|bit)\b", 20.0),
        (r"\b(?:somewhat)\b", 30.0),
        (r"\b(?:half(?:way)?|around)\b", 180.0),
    )
    for pattern, degrees in qualitative_turns:
        if re.search(pattern, text):
            return degrees

    if re.search(r"\b(?:turn|rotate|spin)\b", text):
        return 90.0
    return None


def _extract_walk_distance(text: str) -> float | None:
    directional_distance_patterns = (
        (r"\b(?:walk|move|go)\s+forward\s+(?:for\s+)?(\d+(?:\.\d+)?)\s*(?:meters?|meter|m)\b", 1.0),
        (r"\b(?:walk|move|go)\s+back(?:ward)?\s+(?:for\s+)?(\d+(?:\.\d+)?)\s*(?:meters?|meter|m)\b", -1.0),
        (r"\bforward\s+(?:for\s+)?(\d+(?:\.\d+)?)\s*(?:meters?|meter|m)\b", 1.0),
        (r"\bback(?:ward)?\s+(?:for\s+)?(\d+(?:\.\d+)?)\s*(?:meters?|meter|m)\b", -1.0),
    )
    for pattern, sign in directional_distance_patterns:
        match = re.search(pattern, text)
        if match:
            return _clamp_walk_distance(sign * float(match.group(1)))

    directional_patterns = (
        (r"\b(?:walk|move|go)\s+back(?:ward)?\b", -0.18),
        (r"\b(?:walk|move|go)\s+forward\b", 0.34),
    )
    for pattern, distance in directional_patterns:
        if re.search(pattern, text):
            return distance

    signed_patterns = (
        r"\b(?:walk|move|go)\s+(-?\d+(?:\.\d+)?)\s*(?:meters?|meter|m)\b",
        r"\b(?:walk|move|go)\s+(-?\d+(?:\.\d+)?)\b",
    )
    for pattern in signed_patterns:
        match = re.search(pattern, text)
        if match:
            return _clamp_walk_distance(float(match.group(1)))

    if re.search(r"\b(?:walk|move|go)\b", text):
        return 0.34
    return None


def _coerce_repeat(value: object) -> int:
    if value is None:
        return DEFAULT_REPEAT
    try:
        return max(1, min(5, int(value)))
    except (TypeError, ValueError):
        return DEFAULT_REPEAT


def _coerce_duration(value: object) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    return round(max(0.5, min(8.0, duration)), 2)


def _coerce_turn_degrees(value: object, intent: str) -> float | None:
    if intent not in {"turn_left", "turn_right"}:
        return None
    if value in (None, "", "null"):
        return 90.0
    try:
        degrees = float(value)
    except (TypeError, ValueError):
        return 90.0
    return _clamp_turn_degrees(degrees)


def _coerce_walk_distance(value: object, intent: str) -> float | None:
    if intent not in {"walk_forward", "step_back"}:
        return None
    if value in (None, "", "null"):
        return 0.34 if intent == "walk_forward" else -0.18
    try:
        distance = float(value)
    except (TypeError, ValueError):
        return 0.34 if intent == "walk_forward" else -0.18
    return _clamp_walk_distance(distance)


def _clamp_turn_degrees(value: float) -> float:
    return round(max(1.0, min(360.0, abs(value))), 2)


def _clamp_walk_distance(value: float) -> float:
    return round(max(-20.0, min(20.0, value)), 2)


def _resolve_walk_distance_for_intent(intent: str, walk_distance_m: float | None) -> float | None:
    if intent == "walk_forward":
        if walk_distance_m is None:
            return 0.34
        return walk_distance_m
    if intent == "step_back":
        if walk_distance_m is None:
            return -0.18
        return walk_distance_m if walk_distance_m < 0 else -walk_distance_m
    return None


def _coerce_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.65
    return round(max(0.0, min(1.0, confidence)), 2)


def _actions_signature(actions: list[ParsedAction]) -> list[str]:
    signature: list[str] = []
    for action in actions:
        if action.intent in {"turn_left", "turn_right"}:
            signature.append(f"{action.intent}:{action.turn_degrees}")
        elif action.intent in {"walk_forward", "step_back"}:
            signature.append(f"{action.intent}:{action.walk_distance_m}")
        else:
            signature.append(action.intent)
    return signature


def _action_intents(actions: list[ParsedAction]) -> list[str]:
    return [action.intent for action in actions]


def _action_intent_prefix(fallback_actions: list[ParsedAction], llm_actions: list[ParsedAction]) -> bool:
    fallback_intents = _action_intents(fallback_actions)
    llm_intents = _action_intents(llm_actions)
    if len(fallback_intents) >= len(llm_intents):
        return False
    return llm_intents[: len(fallback_intents)] == fallback_intents


def _enrich_ollama_actions_from_text(actions: list[ParsedAction], raw_text: str) -> list[ParsedAction]:
    if not actions:
        return actions

    segments = _split_segments(_normalize_text(raw_text))
    enriched: list[ParsedAction] = []
    for index, action in enumerate(actions):
        segment = segments[index] if index < len(segments) else _normalize_text(raw_text)
        walk_distance = action.walk_distance_m
        if (
            action.intent in {"walk_forward", "step_back"}
            and _has_explicit_walk_distance(segment)
            and _is_default_walk_distance(action.intent, action.walk_distance_m)
        ):
            inferred_distance = _extract_walk_distance(segment)
            if inferred_distance is not None:
                walk_distance = _resolve_walk_distance_for_intent(action.intent, inferred_distance)

        turn_degrees = action.turn_degrees
        if (
            action.intent in {"turn_left", "turn_right"}
            and _has_explicit_turn_amount(segment)
            and _is_default_turn_degrees(action.turn_degrees)
        ):
            inferred_turn = _extract_turn_degrees(segment)
            if inferred_turn is not None:
                turn_degrees = inferred_turn

        enriched.append(
            ParsedAction(
                intent=action.intent,
                duration_s=action.duration_s,
                repeat=action.repeat,
                turn_degrees=turn_degrees,
                walk_distance_m=walk_distance,
            )
        )
    return enriched


def _normalize_ollama_walk_actions(actions: list[ParsedAction], raw_text: str) -> list[ParsedAction]:
    segments = _split_segments(_normalize_text(raw_text))
    expected_walks: list[tuple[str, float | None]] = []
    for segment in segments:
        parsed = _parse_segment_actions(segment)
        for action in parsed:
            if action.intent in {"walk_forward", "step_back"}:
                expected_walks.append((action.intent, action.walk_distance_m))

    if not expected_walks:
        return actions

    normalized: list[ParsedAction] = []
    walk_index = 0
    for action in actions:
        if action.intent not in {"walk_forward", "step_back"}:
            normalized.append(action)
            continue
        if walk_index >= len(expected_walks):
            continue
        expected_intent, expected_distance = expected_walks[walk_index]
        normalized.append(
            ParsedAction(
                intent=expected_intent,
                duration_s=action.duration_s,
                repeat=action.repeat,
                turn_degrees=action.turn_degrees,
                walk_distance_m=expected_distance if expected_distance is not None else action.walk_distance_m,
            )
        )
        walk_index += 1

    while walk_index < len(expected_walks):
        expected_intent, expected_distance = expected_walks[walk_index]
        normalized.append(
            ParsedAction(
                intent=expected_intent,
                duration_s=None,
                repeat=DEFAULT_REPEAT,
                turn_degrees=None,
                walk_distance_m=expected_distance,
            )
        )
        walk_index += 1

    return normalized


def _has_explicit_walk_distance(text: str) -> bool:
    patterns = (
        r"\b(?:walk|move|go)\s+(?:forward|back|backward)\s+(?:for\s+)?-?\d+(?:\.\d+)?\s*(?:meters?|meter|m)\b",
        r"\b(?:forward|back|backward)\s+(?:for\s+)?-?\d+(?:\.\d+)?\s*(?:meters?|meter|m)\b",
        r"\b(?:walk|move|go)\s+-?\d+(?:\.\d+)?\s*(?:meters?|meter|m)\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _has_explicit_turn_amount(text: str) -> bool:
    patterns = (
        r"\b\d+(?:\.\d+)?\s*(?:degrees?|deg)\b",
        r"\b(?:tiny|slightly|a\s+little|a\s+bit|somewhat|halfway around|around)\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _is_default_walk_distance(intent: str, walk_distance_m: float | None) -> bool:
    default = 0.34 if intent == "walk_forward" else -0.18
    return walk_distance_m is None or abs(walk_distance_m - default) < 1e-9


def _is_default_turn_degrees(turn_degrees: float | None) -> bool:
    return turn_degrees is None or abs(turn_degrees - 90.0) < 1e-9


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def _salvage_payload(text: str) -> dict[str, object] | None:
    actions: list[dict[str, object]] = []
    for match in re.finditer(r'"intent"\s*:\s*"([^"]+)"', text):
        intent = match.group(1).strip()
        if intent in SUPPORTED_INTENTS:
            turn_degrees = 90.0 if intent in {"turn_left", "turn_right"} else None
            walk_distance_m = 0.34 if intent == "walk_forward" else -0.18 if intent == "step_back" else None
            actions.append(
                {
                    "intent": intent,
                    "duration_s": None,
                    "repeat": 1,
                    "turn_degrees": turn_degrees,
                    "walk_distance_m": walk_distance_m,
                }
            )

    if not actions:
        return None

    confidence_match = re.search(r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    confidence = float(confidence_match.group(1)) if confidence_match else 0.65
    return {"actions": actions, "confidence": confidence}
