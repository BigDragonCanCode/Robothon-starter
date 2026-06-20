# Prompt2Action

## Project name

Prompt2Action

## Robot platform

FF Master humanoid using `assets/Master/scene.xml`

## Task goal

Prompt2Action is a natural-language robot demo for FF Robothon. The goal is to let K-12 students and other beginners interact with an FF humanoid robot without needing any programming experience. A student can type a normal request such as `turn left and walk forward`, and the system translates that request into a safe, ordered action sequence the robot can perform.

This is designed around the idea of FF robots being useful in education settings. Students should be able to explore robot behavior immediately, observe how instructions map to actions, and build intuition for robot logic before they learn formal programming.

## Technical approach

- FF Master motion library with bounded scripted primitives for `wave`, `walk_forward`, `step_back`, `turn_left`, `turn_right`, `bow`, `stop`, and `idle`
- Natural-language parser that prefers local `Ollama` with `llama3.2:3b`
- Deterministic fallback parser that still handles chained commands when Ollama is unavailable or unreliable
- Ordered multi-action execution, so one prompt can become a sequence such as `turn_left -> walk_forward`
- Shared runtime for interactive use, batch playback, trajectory logging, and optional video recording

## Core features

- Interactive CLI prompt with the MuJoCo viewer kept open until `quit`
- Kids can command the robot in plain language instead of code
- One input can trigger multiple ordered actions
- Fixed action vocabulary for reproducible judging and safer behavior
- Optional `--batch-file` mode for scripted demo playback
- Optional `--record-video` export to `outputs/prompt2action/`
- Stable JSON session summary with parsed action sequences and trajectory samples

## Educational value

- Low barrier to entry: students can try robot control immediately with natural language
- Clear action grounding: open-ended wording is reduced to a compact motion vocabulary
- Good bridge to computational thinking: students can start with everyday language, then later learn how that language maps to structured robot commands
- Suitable for classroom demos, guided workshops, and beginner robotics exploration

## Current limitation

Prompt2Action does not yet expose the underlying parsed action sequence in a dedicated teaching UI for students. The runtime prints parsed actions in the terminal, but a clearer learner-facing explanation layer is still a planned improvement.

## Future improvements

- Add a student-facing panel that shows the underlying parsed action sequence step by step
- Add richer explanations for why a prompt mapped to a particular action list
- Expand the educational flow with lesson modes, example prompts, and guided challenges
- Replace showcase motions with stronger closed-loop locomotion and balance control
- Add richer multimodal input such as voice or classroom tablet interaction

## Setup

1. Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Optional Ollama setup:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
ollama run llama3.2:3b "reply with json only"
```

3. Confirm the model asset exists:

```bash
ls assets/Master/scene.xml
```

## Run instructions

Interactive viewer mode:

```bash
python3 submissions/prompt2action/run_language_demo.py
```

Interactive mode with deterministic fallback only:

```bash
python3 submissions/prompt2action/run_language_demo.py --no-llm
```

Batch playback with recording:

```bash
python3 submissions/prompt2action/run_language_demo.py \
  --no-llm \
  --batch-file submissions/prompt2action/demo_commands.txt \
  --record-video
```

## Example prompts

- `walk`
- `turn left and walk forward`
- `wave twice then bow`
- `turn right and wave`
- `step back`
- `stop`

## Demo video generation

Create a deterministic artifact with:

```bash
python3 submissions/prompt2action/run_language_demo.py \
  --no-llm \
  --batch-file submissions/prompt2action/demo_commands.txt \
  --record-video \
  --output-dir outputs/prompt2action
```

This writes:

- `outputs/prompt2action/session.mp4` or a GIF fallback
- `outputs/prompt2action/session_summary.json`

## Registration placeholder

Update `registration.json` with the real Robothon UUID before submitting the pull request.
