from __future__ import annotations

import argparse
import json
import select
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import imageio.v3 as iio
    import mujoco
    import mujoco.viewer
except ImportError as exc:
    raise SystemExit(
        "Missing demo dependency. Install with:\n"
        "  python3 -m pip install -r requirements.txt\n\n"
        f"Original error: {exc}"
    ) from exc

from .intents import CommandTranslator, HELP_TEXT, ParsedAction, ParsedCommand
from .monitor import format_diagnostic, format_parsed_command
from .motions import (
    MOTION_LIBRARY,
    RobotState,
    apply_motion_frame,
    apply_neutral_pose,
    capture_joint_pose,
    list_motion_primitives,
    motion_duration,
    resolve_final_state,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "assets" / "Master" / "scene.xml"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "prompt2action"
DEFAULT_SUMMARY = "session_summary.json"
DEFAULT_VIDEO = "session.mp4"


@dataclass(slots=True)
class CommandExecution:
    raw_text: str
    actions: list[dict[str, object]]
    source: str
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_text": self.raw_text,
            "actions": self.actions,
            "source": self.source,
            "confidence": self.confidence,
        }


class FFMasterDemoSession:
    def __init__(
        self,
        *,
        model_path: Path,
        fps: int,
        width: int,
        height: int,
        record_video: bool,
        output_dir: Path,
    ) -> None:
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.fps = fps
        self.width = width
        self.height = height
        self.output_dir = output_dir
        self.record_video = record_video
        self.renderer = mujoco.Renderer(self.model, width=width, height=height) if record_video else None
        self.state = RobotState()
        self.session_time_s = 0.0
        self.frames: list[np.ndarray] = []
        self.command_log: list[dict[str, object]] = []
        self.trajectory: list[dict[str, object]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        apply_neutral_pose(self.model, self.data, self.state)
        self._record_sample("idle")

    def available_motions(self) -> dict[str, object]:
        return {name: primitive.default_duration_s for name, primitive in list_motion_primitives().items()}

    def execute(self, command: ParsedCommand, viewer: object | None = None) -> CommandExecution | None:
        if not command.actions:
            return None

        action_summaries: list[dict[str, object]] = []
        for index, action in enumerate(command.actions):
            is_last_action = index == len(command.actions) - 1
            action_summaries.append(self._execute_action(action, viewer, settle_after=is_last_action))

        execution = CommandExecution(
            raw_text=command.raw_text,
            actions=action_summaries,
            source=command.source,
            confidence=command.confidence,
        )
        self.command_log.append(execution.to_dict())
        return execution

    def _execute_action(
        self,
        action: ParsedAction,
        viewer: object | None,
        *,
        settle_after: bool,
    ) -> dict[str, object]:
        duration_s = motion_duration(
            action.intent,
            action.duration_s,
            action.turn_degrees,
            action.walk_distance_m,
        )
        repeats = max(1, action.repeat)
        for _ in range(repeats):
            source_pose = capture_joint_pose(self.model, self.data)
            start_state = RobotState(
                self.state.base_x,
                self.state.base_y,
                self.state.base_z,
                self.state.yaw,
            )
            end_state = resolve_final_state(
                action.intent,
                start_state,
                action.turn_degrees,
                action.walk_distance_m,
            )
            total_frames = max(1, int(round(duration_s * self.fps)))
            for frame_idx in range(total_frames):
                elapsed_s = frame_idx / self.fps
                apply_motion_frame(
                    self.model,
                    self.data,
                    action.intent,
                    start_state,
                    end_state,
                    elapsed_s,
                    duration_s,
                    source_pose=source_pose,
                )
                self.session_time_s += 1.0 / self.fps
                self._record_sample(action.intent, frame_idx)
                if self.renderer is not None:
                    self.renderer.update_scene(self.data)
                    self.frames.append(self.renderer.render().copy())
                if viewer is not None:
                    try:
                        viewer.sync()
                    except Exception:
                        viewer = None
                time.sleep(1.0 / self.fps) if viewer is not None else None
            self.state = end_state
        if settle_after:
            self._settle_to_idle(viewer)
        return {
            "intent": action.intent,
            "duration_s": duration_s,
            "repeat": repeats,
            "turn_degrees": action.turn_degrees,
            "walk_distance_m": action.walk_distance_m,
        }

    def _settle_to_idle(self, viewer: object | None) -> None:
        duration_s = 0.6
        total_frames = max(1, int(round(duration_s * self.fps)))
        source_pose = capture_joint_pose(self.model, self.data)
        for frame_idx in range(total_frames):
            elapsed_s = frame_idx / self.fps
            apply_motion_frame(
                self.model,
                self.data,
                "idle",
                self.state,
                self.state,
                elapsed_s,
                duration_s,
                source_pose=source_pose,
            )
            self.session_time_s += 1.0 / self.fps
            self._record_sample("idle", frame_idx)
            if self.renderer is not None:
                self.renderer.update_scene(self.data)
                self.frames.append(self.renderer.render().copy())
            if viewer is not None:
                try:
                    viewer.sync()
                except Exception:
                    viewer = None
            time.sleep(1.0 / self.fps) if viewer is not None else None

    def finalize(self) -> dict[str, object]:
        summary_path = self.output_dir / DEFAULT_SUMMARY
        video_path = self.output_dir / DEFAULT_VIDEO
        final_pelvis = body_position(self.model, self.data, "pelvis")
        summary = {
            "project": "Prompt2Action",
            "robot_platform": "FF Master humanoid",
            "model": str(self.model_path),
            "commands_executed": self.command_log,
            "trajectory_samples": self.trajectory,
            "video": str(video_path) if self.record_video else None,
            "summary_path": str(summary_path),
            "record_video": self.record_video,
            "final_pelvis_pos": final_pelvis,
            "session_time_s": round(self.session_time_s, 3),
            "supported_intents": sorted(MOTION_LIBRARY.keys()),
        }

        if self.record_video and self.frames:
            try:
                iio.imwrite(video_path, np.asarray(self.frames), fps=self.fps, codec="libx264")
            except Exception as exc:
                fallback = video_path.with_suffix(".gif")
                iio.imwrite(fallback, np.asarray(self.frames), fps=self.fps)
                summary["video"] = str(fallback)
                summary["video_fallback_reason"] = str(exc)

        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _record_sample(self, intent: str, frame_idx: int = 0) -> None:
        sample_stride = max(1, self.fps // 5)
        if frame_idx % sample_stride != 0:
            return
        self.trajectory.append(
            {
                "time_s": round(self.session_time_s, 3),
                "intent": intent,
                "pelvis_pos": body_position(self.model, self.data, "pelvis"),
                "yaw_rad": round(self.state.yaw, 4),
                "qpos_head": self.data.qpos[:10].round(4).tolist(),
            }
        )


def body_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> list[float]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Missing body in MJCF: {body_name}")
    return data.xpos[body_id].copy().round(5).tolist()


def load_batch_commands(batch_file: Path) -> list[str]:
    commands: list[str] = []
    for line in batch_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            commands.append(stripped)
    return commands


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the FF Master language-controlled MuJoCo demo."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--ollama-model", default="llama3.2:3b")
    parser.add_argument("--no-llm", action="store_true", help="Force deterministic fallback parsing.")
    parser.add_argument("--batch-file", type=Path, help="Run commands from a text file instead of interactive input.")
    parser.add_argument("--record-video", action="store_true", help="Record the rendered session to the output directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--headless", action="store_true", help="Disable the live MuJoCo viewer.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    translator = CommandTranslator(
        ollama_model=args.ollama_model,
        use_llm=not args.no_llm,
    )
    session = FFMasterDemoSession(
        model_path=args.model_path,
        fps=args.fps,
        width=args.width,
        height=args.height,
        record_video=args.record_video,
        output_dir=args.output_dir,
    )

    if args.batch_file is not None:
        commands = load_batch_commands(args.batch_file)
        summary = run_batch_session(session, translator, commands)
        print(json.dumps(summary, indent=2))
        return 0

    if args.headless:
        summary = run_interactive_headless(session, translator)
        print(json.dumps(summary, indent=2))
        return 0

    try:
        with mujoco.viewer.launch_passive(session.model, session.data) as viewer:
            print(HELP_TEXT)
            print("Type a command, then press Enter. Type 'quit' to exit.")
            summary = run_interactive_session(session, translator, viewer)
    except Exception as exc:
        print(f"Viewer unavailable, falling back to headless mode: {exc}", file=sys.stderr)
        summary = run_interactive_headless(session, translator)

    print(json.dumps(summary, indent=2))
    return 0


def run_batch_session(
    session: FFMasterDemoSession,
    translator: CommandTranslator,
    commands: list[str],
) -> dict[str, object]:
    for raw_command in commands:
        parsed = translator.parse(raw_command)
        if not parsed.actions:
            print(f"[ignored] {raw_command}: {parsed.message}")
            continue
        print(
            f"[parsed] {format_parsed_command(parsed)}"
        )
        diagnostic = format_diagnostic(parsed)
        if diagnostic:
            print(diagnostic)
        session.execute(parsed, viewer=None)
    return session.finalize()


def run_interactive_headless(
    session: FFMasterDemoSession,
    translator: CommandTranslator,
) -> dict[str, object]:
    print(HELP_TEXT)
    while True:
        try:
            raw_command = input("> ").strip()
        except EOFError:
            break
        if raw_command.lower() in {"quit", "exit"}:
            break
        _handle_command(raw_command, session, translator, viewer=None)
    return session.finalize()


def run_interactive_session(
    session: FFMasterDemoSession,
    translator: CommandTranslator,
    viewer: object,
) -> dict[str, object]:
    while _viewer_running(viewer):
        raw_command = _prompt_line("> ", viewer)
        if raw_command is None:
            break
        raw_command = raw_command.strip()
        if not raw_command:
            continue
        if raw_command.lower() in {"quit", "exit"}:
            break
        _handle_command(raw_command, session, translator, viewer=viewer)
    return session.finalize()


def _handle_command(
    raw_command: str,
    session: FFMasterDemoSession,
    translator: CommandTranslator,
    viewer: object | None,
) -> None:
    parsed = translator.parse(raw_command)
    if not parsed.actions:
        diagnostic = format_diagnostic(parsed)
        if diagnostic:
            print(diagnostic)
        print(parsed.message or HELP_TEXT)
        return

    print(format_parsed_command(parsed))
    diagnostic = format_diagnostic(parsed)
    if diagnostic:
        print(diagnostic)
    session.execute(parsed, viewer=viewer)


def _prompt_line(prompt: str, viewer: object) -> str | None:
    print(prompt, end="", flush=True)
    while _viewer_running(viewer):
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        if ready:
            line = sys.stdin.readline()
            if line == "":
                return None
            return line.rstrip("\n")
    return None


def _viewer_running(viewer: object) -> bool:
    running = getattr(viewer, "is_running", None)
    if callable(running):
        return bool(running())
    return True
