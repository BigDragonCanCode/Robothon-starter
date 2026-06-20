from __future__ import annotations

from .intents import ParsedCommand


def format_parsed_command(command: ParsedCommand) -> str:
    actions = []
    for action in command.actions:
        label = action.intent
        if action.turn_degrees is not None:
            label = f"{label}({action.turn_degrees}deg)"
        if action.walk_distance_m is not None:
            label = f"{label}({action.walk_distance_m}m)"
        actions.append(label)
    return (
        f"actions={actions} "
        f"source={command.source} confidence={command.confidence:.2f}"
    )


def format_diagnostic(command: ParsedCommand) -> str | None:
    return command.diagnostics
