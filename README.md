# AI Research Agent System

An end-to-end automated AI research paper generation system using a multi-agent pipeline.

## Architecture

The system uses four specialized agents in sequence:

1. **Ideation Agent** - Generates and evaluates research ideas via literature search
2. **Planning Agent** - Creates detailed experimental plans and methodology
3. **Experiment Agent** - Executes experiments in sandboxed environments
4. **Writing Agent** - Composes full research papers in LaTeX

## Setup

```bash
pip install -e ".[dev]"
```

## Usage

```python
from ai_research_agent import Project, Orchestrator
from pathlib import Path

project = Project(Path("./my_research_project"))
project.initialize()

orchestrator = Orchestrator(project)
orchestrator.run(topic="novel attention mechanisms for transformers")
```

## Project Structure

Each research project creates a directory with:
- `status.json` - current pipeline status
- `audit.jsonl` - append-only audit log of all tool calls
- `idea/` - ideation artifacts
- `plan/` - planning artifacts
- `experiments/` - experiment code and results
- `writing/` - paper drafts and final PDF
