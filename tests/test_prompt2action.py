from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from submissions.prompt2action.intents import (
    CommandTranslator,
    HELP_TEXT,
    ParsedAction,
    ParsedCommand,
)
from submissions.prompt2action.motions import list_motion_primitives
from submissions.prompt2action.motions import resolve_final_state, RobotState
from submissions.prompt2action.simulator import (
    DEFAULT_MODEL,
    FFMasterDemoSession,
    run_batch_session,
)


class FakeOllamaClient:
    def __init__(self, payload: dict[str, object] | Exception) -> None:
        self.payload = payload

    def is_available(self) -> bool:
        return True

    def translate(self, command: str) -> dict[str, object]:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class BrokenOllamaClient:
    def is_available(self) -> bool:
        return True

    def translate(self, command: str) -> dict[str, object]:
        raise ValueError("malformed output")


class CommandTranslatorTests(unittest.TestCase):
    def test_fallback_maps_known_phrase(self) -> None:
        translator = CommandTranslator(use_llm=False)
        parsed = translator.parse("walk forward for 3 seconds")
        self.assertEqual([action.intent for action in parsed.actions], ["walk_forward"])
        self.assertEqual(parsed.actions[0].duration_s, 3.0)
        self.assertEqual(parsed.actions[0].repeat, 1)
        self.assertEqual(parsed.actions[0].walk_distance_m, 0.34)
        self.assertEqual(parsed.source, "fallback")

    def test_fallback_extracts_repeat(self) -> None:
        translator = CommandTranslator(use_llm=False)
        parsed = translator.parse("turn right twice")
        self.assertEqual(parsed.actions[0].intent, "turn_right")
        self.assertEqual(parsed.actions[0].repeat, 2)
        self.assertEqual(parsed.actions[0].turn_degrees, 90.0)

    def test_fallback_defaults_turn_to_ninety_degrees(self) -> None:
        translator = CommandTranslator(use_llm=False)
        parsed = translator.parse("turn left")
        self.assertEqual(parsed.actions[0].turn_degrees, 90.0)

    def test_fallback_extracts_numeric_turn_degrees(self) -> None:
        translator = CommandTranslator(use_llm=False)
        parsed = translator.parse("turn left 45 degrees")
        self.assertEqual(parsed.actions[0].turn_degrees, 45.0)

    def test_fallback_extracts_qualitative_small_turn(self) -> None:
        translator = CommandTranslator(use_llm=False)
        parsed = translator.parse("turn a little right")
        self.assertEqual(parsed.actions[0].intent, "turn_right")
        self.assertEqual(parsed.actions[0].turn_degrees, 15.0)

    def test_fallback_extracts_qualitative_small_turn_to_the_left(self) -> None:
        translator = CommandTranslator(use_llm=False)
        parsed = translator.parse("turn a little to the left")
        self.assertEqual(parsed.actions[0].intent, "turn_left")
        self.assertEqual(parsed.actions[0].turn_degrees, 15.0)

    def test_fallback_maps_bare_walk(self) -> None:
        translator = CommandTranslator(use_llm=False)
        parsed = translator.parse("walk")
        self.assertEqual(parsed.actions[0].intent, "walk_forward")
        self.assertEqual(parsed.actions[0].walk_distance_m, 0.34)

    def test_fallback_understands_walk_backward(self) -> None:
        translator = CommandTranslator(use_llm=False)
        parsed = translator.parse("walk backward")
        self.assertEqual(parsed.actions[0].intent, "step_back")
        self.assertEqual(parsed.actions[0].walk_distance_m, -0.18)

    def test_fallback_extracts_signed_walk_distance(self) -> None:
        translator = CommandTranslator(use_llm=False)
        parsed = translator.parse("walk -0.5 meters")
        self.assertEqual(parsed.actions[0].intent, "walk_forward")
        self.assertEqual(parsed.actions[0].walk_distance_m, -0.5)

    def test_fallback_parses_forward_then_backward_with_distances(self) -> None:
        translator = CommandTranslator(use_llm=False)
        parsed = translator.parse("i want to first walk forward for 10 meter, and then backward for 5 meter")
        self.assertEqual([action.intent for action in parsed.actions], ["walk_forward", "step_back"])
        self.assertEqual(parsed.actions[0].walk_distance_m, 10.0)
        self.assertEqual(parsed.actions[1].walk_distance_m, -5.0)

    def test_fallback_parses_multi_action_sequence(self) -> None:
        translator = CommandTranslator(use_llm=False)
        parsed = translator.parse("turn left and walk forward")
        self.assertEqual([action.intent for action in parsed.actions], ["turn_left", "walk_forward"])

    def test_invalid_ollama_output_falls_back_cleanly(self) -> None:
        translator = CommandTranslator(
            use_llm=True,
            ollama_client=FakeOllamaClient({"actions": [{"intent": "dance", "repeat": 3}]}),
        )
        parsed = translator.parse("wave three times")
        self.assertEqual(parsed.actions[0].intent, "wave")
        self.assertEqual(parsed.source, "fallback")
        self.assertEqual(parsed.actions[0].repeat, 3)
        self.assertIsNone(parsed.diagnostics)

    def test_ollama_disagreement_uses_fallback(self) -> None:
        translator = CommandTranslator(
            use_llm=True,
            ollama_client=FakeOllamaClient(
                {"actions": [{"intent": "turn_left", "repeat": 1}], "confidence": 0.9}
            ),
        )
        parsed = translator.parse("walk forward")
        self.assertEqual(parsed.actions[0].intent, "walk_forward")
        self.assertEqual(parsed.source, "fallback")
        self.assertIn("Ollama disagreement", parsed.diagnostics or "")

    def test_ollama_multi_action_is_used_when_it_matches_fallback(self) -> None:
        translator = CommandTranslator(
            use_llm=True,
            ollama_client=FakeOllamaClient(
                {
                    "actions": [
                        {"intent": "turn_left", "repeat": 1},
                        {"intent": "walk_forward", "repeat": 1},
                    ],
                    "confidence": 0.91,
                }
            ),
        )
        parsed = translator.parse("turn left and walk forward")
        self.assertEqual([action.intent for action in parsed.actions], ["turn_left", "walk_forward"])
        self.assertEqual(parsed.source, "ollama")

    def test_ollama_turn_parameters_are_used_when_intents_match(self) -> None:
        translator = CommandTranslator(
            use_llm=True,
            ollama_client=FakeOllamaClient(
                {
                    "actions": [
                        {"intent": "turn_left", "repeat": 1, "turn_degrees": 12},
                    ],
                    "confidence": 0.94,
                }
            ),
        )
        parsed = translator.parse("turn a little to the left")
        self.assertEqual(parsed.source, "ollama")
        self.assertEqual(parsed.actions[0].intent, "turn_left")
        self.assertEqual(parsed.actions[0].turn_degrees, 12.0)

    def test_ollama_walk_parameters_are_used_when_intents_match(self) -> None:
        translator = CommandTranslator(
            use_llm=True,
            ollama_client=FakeOllamaClient(
                {
                    "actions": [
                        {"intent": "walk_forward", "repeat": 1, "walk_distance_m": -0.5},
                    ],
                    "confidence": 0.94,
                }
            ),
        )
        parsed = translator.parse("walk -0.5 meters")
        self.assertEqual(parsed.source, "ollama")
        self.assertEqual(parsed.actions[0].walk_distance_m, -0.5)

    def test_ollama_walk_distances_are_recovered_from_text_when_missing(self) -> None:
        translator = CommandTranslator(
            use_llm=True,
            ollama_client=FakeOllamaClient(
                {
                    "actions": [
                        {"intent": "walk_forward", "repeat": 1},
                        {"intent": "step_back", "repeat": 1},
                    ],
                    "confidence": 0.95,
                }
            ),
        )
        parsed = translator.parse("walk forward for 5m and back for 2m")
        self.assertEqual(parsed.source, "ollama")
        self.assertEqual([action.intent for action in parsed.actions], ["walk_forward", "step_back"])
        self.assertEqual(parsed.actions[0].walk_distance_m, 5.0)
        self.assertEqual(parsed.actions[1].walk_distance_m, -2.0)

    def test_ollama_bad_walk_clause_alignment_is_normalized(self) -> None:
        translator = CommandTranslator(
            use_llm=True,
            ollama_client=FakeOllamaClient(
                {
                    "actions": [
                        {"intent": "walk_forward", "repeat": 1, "walk_distance_m": 2},
                        {"intent": "walk_forward", "repeat": 1, "walk_distance_m": -1},
                        {"intent": "step_back", "repeat": 1, "walk_distance_m": -2},
                    ],
                    "confidence": 0.95,
                }
            ),
        )
        parsed = translator.parse("walk forward for 2m, back for 1 meter")
        self.assertEqual(parsed.source, "ollama")
        self.assertEqual([action.intent for action in parsed.actions], ["walk_forward", "step_back"])
        self.assertEqual(parsed.actions[0].walk_distance_m, 2.0)
        self.assertEqual(parsed.actions[1].walk_distance_m, -1.0)

    def test_ollama_sequence_is_used_when_fallback_only_finds_prefix(self) -> None:
        translator = CommandTranslator(
            use_llm=True,
            ollama_client=FakeOllamaClient(
                {
                    "actions": [
                        {"intent": "walk_forward", "repeat": 1, "walk_distance_m": 10},
                        {"intent": "step_back", "repeat": 1, "walk_distance_m": -5},
                    ],
                    "confidence": 0.95,
                }
            ),
        )
        parsed = translator.parse("i want to first walk forward for 10 meter, and then backward for 5 meter")
        self.assertEqual(parsed.source, "ollama")
        self.assertEqual([action.intent for action in parsed.actions], ["walk_forward", "step_back"])
        self.assertEqual(parsed.actions[0].walk_distance_m, 10.0)
        self.assertEqual(parsed.actions[1].walk_distance_m, -5.0)

    def test_unsupported_text_returns_help(self) -> None:
        translator = CommandTranslator(use_llm=False)
        parsed = translator.parse("do a backflip")
        self.assertEqual(parsed.actions, [])
        self.assertEqual(parsed.message, HELP_TEXT)

    def test_ollama_failure_is_silent_when_fallback_succeeds(self) -> None:
        translator = CommandTranslator(use_llm=True, ollama_client=BrokenOllamaClient())
        parsed = translator.parse("turn left and walk forward")
        self.assertEqual([action.intent for action in parsed.actions], ["turn_left", "walk_forward"])
        self.assertIsNone(parsed.diagnostics)


class MotionSessionTests(unittest.TestCase):
    def test_motion_library_contains_expected_intents(self) -> None:
        primitives = list_motion_primitives()
        self.assertEqual(
            set(primitives),
            {"wave", "walk_forward", "step_back", "turn_left", "turn_right", "bow", "stop", "idle"},
        )

    def test_resolve_final_state_uses_requested_turn_degrees(self) -> None:
        start = RobotState()
        end = resolve_final_state("turn_left", start, 45.0)
        self.assertAlmostEqual(end.yaw, np.deg2rad(45.0))

    def test_resolve_final_state_uses_signed_walk_distance(self) -> None:
        start = RobotState()
        end = resolve_final_state("walk_forward", start, walk_distance_m=-0.5)
        self.assertAlmostEqual(end.base_x, -0.5)

    def test_resolve_final_state_walks_in_current_heading(self) -> None:
        start = RobotState(yaw=np.deg2rad(90.0))
        end = resolve_final_state("walk_forward", start, walk_distance_m=1.0)
        self.assertAlmostEqual(end.base_x, 0.0, places=6)
        self.assertAlmostEqual(end.base_y, 1.0, places=6)

    def test_stop_and_idle_return_neutral_without_exceptions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = FFMasterDemoSession(
                model_path=DEFAULT_MODEL,
                fps=20,
                width=320,
                height=240,
                record_video=False,
                output_dir=Path(tmp),
            )
            for intent in ("idle", "stop"):
                result = session.execute(
                    ParsedCommand(
                        actions=[ParsedAction(intent=intent)],
                        raw_text=intent,
                        confidence=1.0,
                        source="test",
                    )
                )
                self.assertIsNotNone(result)
                self.assertTrue(np.isfinite(session.data.qpos).all())
                self.assertTrue(np.allclose(session.data.qvel, 0.0))

    def test_repeated_execution_stays_finite_and_logs_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = FFMasterDemoSession(
                model_path=DEFAULT_MODEL,
                fps=15,
                width=320,
                height=240,
                record_video=False,
                output_dir=Path(tmp),
            )
            translator = CommandTranslator(use_llm=False)
            summary = run_batch_session(
                session,
                translator,
                ["turn left and walk forward", "wave twice", "bow", "stop"],
            )
            self.assertGreaterEqual(len(summary["commands_executed"]), 4)
            self.assertTrue(np.isfinite(session.data.qpos).all())
            self.assertTrue(np.isfinite(session.data.qvel).all())
            self.assertGreater(len(summary["trajectory_samples"]), 0)
            self.assertTrue(Path(summary["summary_path"]).exists())
            self.assertEqual(
                summary["commands_executed"][0]["actions"][0]["intent"],
                "turn_left",
            )


if __name__ == "__main__":
    unittest.main()
