# Research Agent System

A modular framework for automating the research workflow: from literature review and hypothesis generation to experimental validation and paper composition.

## Overview

This system implements a multi-agent architecture where specialized components collaborate through a shared file system to produce reproducible research artifacts. Each agent operates independently, consuming structured inputs and producing validated outputs that feed into downstream stages.

The design prioritizes:
- **Transparency**: Every operation is logged to an append-only audit trail
- **Reproducibility**: All artifacts are versioned and validated against schemas
- **Extensibility**: New agents and tools can be added without modifying existing code
- **Verification**: Outputs are mechanically checked (number verification, citation consistency, LaTeX compilation)

## Architecture

The system consists of four sequential agents connected by a state machine:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Ideation   │────▶│  Planning   │────▶│  Experiment │────▶│   Writing   │
│   Agent     │     │   Agent     │     │   Agent     │     │   Agent     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                    │                    │                    │
   idea/                 plan/              experiments/          writing/
   ├── proposal.md       ├── plan.md        ├── report.md        ├── paper.tex
   └── ideation.json     └── plan.json      ├── report.json      └── paper.pdf
                                             └── code/
```

### Agent Responsibilities

**Ideation Agent**
- Surveys literature (arXiv, Semantic Scholar) to identify gaps
- Generates binary-falsifiable hypotheses with quantifiable success criteria
- Performs novelty kill searches to verify research originality
- Produces 3-5 candidate ideas with failure mode analysis (≥3 scenarios per idea)
- Selects the best idea or rejects all if none meet thresholds

**Planning Agent**
- Translates research hypotheses into executable experiment plans
- Specifies datasets, baselines, evaluation metrics, and ablation studies
- Locks hyperparameters and random seeds before execution
- Assesses technical risks with mitigation strategies

**Experiment Agent**
- Generates Python code from plan specifications
- Executes in Docker sandboxes with resource limits (512MB RAM, 120s timeout)
- Retries failed steps up to 3 times with exponential backoff
- Aggregates results across random seeds (mean ± std)
- Performs statistical significance testing (t-test)

**Writing Agent**
- Composes LaTeX papers in ACL/ICML style
- Verifies every number against experiment reports
- Generates BibTeX citations from literature survey
- Produces camera-ready PDFs (if pdflatex available)

## Installation

```bash
# Clone the repository
git clone https://github.com/THRIDX/research-agent-system.git
cd research-agent-system

# Install with dependencies
pip install -e ".[dev]"

# Verify installation
python -c "from research_agent import Project; print('OK')"
```

### Requirements

- Python 3.11+
- Docker (optional, for sandboxed execution)
- pdflatex (optional, for PDF generation)

## Quick Start

```python
from pathlib import Path
from research_agent import Project, Orchestrator

# Initialize a research project
project = Project(Path("./my_research"))
project.initialize()

# Run the full pipeline
orchestrator = Orchestrator(project, use_docker=True)
orchestrator.run(topic="parameter-efficient fine-tuning for vision transformers")
```

## Project Directory Structure

Each research project maintains the following structure:

```
my_research/
├── status.json          # Current pipeline state
├── audit.jsonl          # Append-only log of all operations
├── idea/
│   ├── proposal.md      # Human-readable research proposal
│   └── ideation.json    # Structured candidate ideas
├── plan/
│   ├── plan.md          # Experimental design document
│   └── plan.json        # Machine-readable plan
├── experiments/
│   ├── code/            # Generated Python scripts
│   ├── report.md        # Experiment results
│   └── report.json      # Structured metrics
└── writing/
    ├── paper.tex        # LaTeX source
    ├── paper.pdf        # Compiled PDF
    └── paper.md         # Markdown version
```

## Configuration

Agent behavior can be customized via environment variables:

```bash
# Execution settings
export USE_DOCKER=true              # Use Docker sandbox (default: false)
export EXPERIMENT_TIMEOUT=120       # Seconds per step (default: 120)
export MAX_RETRIES=3                # Retry attempts for failed steps

# Search settings
export ARXIV_MAX_RESULTS=20         # Papers to fetch from arXiv
export SEMANTIC_SCHOLAR_MAX=15      # Papers from Semantic Scholar

# Selection thresholds
export MIN_FEASIBILITY=0.6          # Minimum feasibility score (0-1)
export MIN_IMPACT=0.5               # Minimum impact score (0-1)
export MIN_NOVELTY=0.7              # Minimum novelty score (0-1)
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific agent tests
pytest tests/test_ideation.py -v
pytest tests/test_planning.py -v
pytest tests/test_experiment.py -v
pytest tests/test_writing.py -v

# Run integration tests
pytest tests/test_integration.py -v
```

## Design Principles

### File System as Communication Medium

Agents do not communicate directly. Instead, they read and write files to a shared project directory. This design:
- Enables inspection of intermediate states
- Supports resumability after interruptions
- Allows human intervention at any stage

### Atomic Writes

All file modifications use atomic operations (write to temporary file, then rename) to prevent corruption during crashes.

### Audit Logging

Every tool call is recorded with:
- Timestamp
- Agent name
- Tool name and inputs
- Outputs or error details

This enables full traceability of the research process.

### Binary Falsifiable Hypotheses

The system enforces that all research hypotheses must be testable with a YES/NO outcome:

✓ Valid: "Fine-tuning with LoRA on 1000 examples achieves >85% accuracy"

✗ Invalid: "Our method improves performance" (no threshold)

### Novelty Kill Search

Before accepting any hypothesis, the system searches for existing work using multiple query strategies:
1. Exact method combination
2. Claimed improvement range
3. Problem + technique overlap

If no differentiation is found, the idea is rejected.

## Extending the System

### Adding a New Tool

Create a new module in `src/research_agent/tools/`:

```python
# src/research_agent/tools/my_tool.py
from pydantic import BaseModel

class MyToolResult(BaseModel):
    value: float

def my_tool(input_data: str) -> MyToolResult:
    # Implementation
    return MyToolResult(value=42.0)
```

### Adding a New Agent

Subclass `BaseAgent` and implement the `run()` method:

```python
from research_agent.agents.base import BaseAgent
from research_agent.models.common import ProjectStatus

class MyAgent(BaseAgent):
    agent_status = ProjectStatus.MY_STAGE
    
    def run(self) -> MyOutput:
        # Implementation
        pass
```

## Limitations

- **LLM Integration**: Current hypothesis generation uses rule-based templates. Integration with language models would improve idea quality.
- **Execution Environment**: Docker execution requires Docker daemon running. Local execution is less isolated.
- **Citation Management**: BibTeX generation is basic. Manual review of citations is recommended.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{research_agent_system,
  title = {Research Agent System: A Modular Framework for Automated Research Workflows},
  author = {Research Tools Group},
  year = {2024},
  url = {https://github.com/THRIDX/research-agent-system}
}
```

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome. Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

For major changes, please open an issue first to discuss the proposed changes.
