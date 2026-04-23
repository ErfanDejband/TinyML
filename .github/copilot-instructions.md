# TinyML Project — Copilot Instructions

## Context
Read `PROJECT_CONTEXT.md` before any task — it has current status, model specs, and project structure.

## Student Profile
- **Python**: Proficient | **C/C++**: Beginner (learning) | **Goal**: Master TinyML end-to-end
- **Learning style**: Intuitive analogies, worked math examples, ASCII diagrams
- **Pet peeves**: Don't rush topics. Don't give long answers when short ones work.

## Code Style
- Type hints required in Python
- Use `tf_keras` (not `keras`) — tfmot compatibility
- Follow existing patterns in `Phase_1/*.py`
- Explain WHY, not just HOW

## Agents
Use `@orchestrator` as the entry point — it routes to the right specialist:
- `@code-helper` — write, debug, review code
- `@deep-explainer` — math, theory, diagrams, analogies
- `@roadmap-manager` — progress tracking, next steps, validation