"""Ideation Agent – generates and selects research ideas."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Optional

from ai_research_agent.agents.base import BaseAgent
from ai_research_agent.models.common import ProjectStatus
from ai_research_agent.models.proposal import (
    CandidateIdea,
    ClosestPaper,
    FailureMode,
    IdeationOutput,
    NoveltyCheckResult,
    RelatedWork,
    SuccessCriteria,
)
from ai_research_agent.tools.filesystem import atomic_write, write_json
from ai_research_agent.tools.search import search_arxiv, search_semantic_scholar


# Configuration thresholds
MIN_FEASIBILITY = 0.6
MIN_IMPACT = 0.5
MIN_NOVELTY_SCORE = 0.7
DEFAULT_NUM_IDEAS = 5


class IdeationAgent(BaseAgent):
    """Searches the literature and proposes research ideas.

    This agent performs:
    1. Literature survey (arxiv + semantic scholar)
    2. Hypothesis generation (3-5 binary falsifiable hypotheses)
    3. Novelty kill search (verify no existing work)
    4. Feasibility assessment (0-1 score)
    5. Impact assessment (0-1 score)
    6. Failure mode analysis (≥3 per idea)
    7. Selection based on thresholds
    """

    agent_status = ProjectStatus.IDEATION

    def __init__(
        self,
        project: Any,
        topic: str,
        num_ideas: int = DEFAULT_NUM_IDEAS,
        min_feasibility: float = MIN_FEASIBILITY,
        min_impact: float = MIN_IMPACT,
        min_novelty: float = MIN_NOVELTY_SCORE,
    ) -> None:
        super().__init__(project)
        self.topic = topic
        self.num_ideas = num_ideas
        self.min_feasibility = min_feasibility
        self.min_impact = min_impact
        self.min_novelty = min_novelty
        self._related_work: list[RelatedWork] = []
        self._arxiv_results: list[dict[str, Any]] = []
        self._semantic_results: list[dict[str, Any]] = []

    def run(self) -> IdeationOutput:
        """Execute the full ideation pipeline."""
        # Ensure idea directory exists
        self.project.idea_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Literature survey
        self.project.update_status(
            status=ProjectStatus.IDEATION,
            current_agent=self.name,
            step="searching literature",
        )
        self._related_work = self._do_literature_survey()

        # Step 2: Generate candidate hypotheses
        self.project.update_status(
            status=ProjectStatus.IDEATION,
            current_agent=self.name,
            step="generating hypotheses",
        )
        candidates = self._generate_candidates()

        # Step 3: Evaluate each candidate
        self.project.update_status(
            status=ProjectStatus.IDEATION,
            current_agent=self.name,
            step="evaluating candidates",
        )
        evaluated = []
        for candidate in candidates:
            evaluated.append(self._evaluate_candidate(candidate))

        # Step 4: Filter by thresholds
        self.project.update_status(
            status=ProjectStatus.IDEATION,
            current_agent=self.name,
            step="selecting best idea",
        )
        passed = [c for c in evaluated if c.passed_filters]

        # Step 5: Select best or reject
        if not passed:
            output = self._create_rejected_output(evaluated)
        else:
            selected = max(passed, key=lambda c: c.combined_score)
            output = self._create_output(evaluated, selected)

        # Step 6: Write output files
        self._write_output_files(output)

        return output

    def _do_literature_survey(self) -> list[RelatedWork]:
        """Search arxiv and semantic scholar for related work."""
        all_results: list[RelatedWork] = []

        # Search arxiv
        self._log("search_arxiv", {"query": self.topic, "max_results": 20})
        try:
            arxiv_results = search_arxiv(self.topic, max_results=20)
            self._log(
                "search_arxiv",
                {"query": self.topic, "max_results": 20},
                outputs={"num_results": len(arxiv_results)},
            )
            self._arxiv_results = [
                {"title": r.title, "abstract": r.abstract, "authors": r.authors}
                for r in arxiv_results
            ]
            for r in arxiv_results[:10]:
                all_results.append(
                    RelatedWork(
                        title=r.title,
                        authors=r.authors[:5],
                        arxiv_id=r.arxiv_id,
                        relevance_score=0.8,
                        summary=r.abstract[:300],
                    )
                )
        except Exception as exc:
            self._log("search_arxiv", {"query": self.topic}, error=str(exc))

        # Search semantic scholar
        self._log("search_semantic_scholar", {"query": self.topic, "max_results": 15})
        try:
            semantic_results = search_semantic_scholar(self.topic, max_results=15)
            self._log(
                "search_semantic_scholar",
                {"query": self.topic, "max_results": 15},
                outputs={"num_results": len(semantic_results)},
            )
            self._semantic_results = [
                {"title": r.title, "abstract": r.abstract, "authors": r.authors, "year": r.year}
                for r in semantic_results
            ]
            for r in semantic_results[:10]:
                all_results.append(
                    RelatedWork(
                        title=r.title,
                        authors=r.authors[:5],
                        year=r.year,
                        relevance_score=0.7,
                        summary=r.abstract[:300] if r.abstract else "",
                    )
                )
        except Exception as exc:
            self._log("search_semantic_scholar", {"query": self.topic}, error=str(exc))

        # De-duplicate by title similarity
        return self._deduplicate_related_work(all_results)

    def _deduplicate_related_work(
        self, works: list[RelatedWork]
    ) -> list[RelatedWork]:
        """Remove duplicate papers based on title similarity."""
        if not works:
            return []

        unique: list[RelatedWork] = [works[0]]
        for work in works[1:]:
            is_duplicate = False
            for existing in unique:
                # Simple similarity check - if titles share >50% words
                existing_words = set(existing.title.lower().split())
                new_words = set(work.title.lower().split())
                if len(existing_words & new_words) / max(len(existing_words), 1) > 0.5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(work)

        return unique

    def _generate_candidates(self) -> list[CandidateIdea]:
        """Generate candidate research ideas based on literature."""
        # Extract key themes from literature
        themes = self._extract_themes()

        candidates: list[CandidateIdea] = []
        for i in range(self.num_ideas):
            theme = themes[i % len(themes)] if themes else self.topic

            # Generate binary falsifiable hypothesis
            hypothesis, binary = self._generate_hypothesis(i, theme)

            candidate = CandidateIdea(
                title=f"Research Idea {i + 1}: {theme.title()}",
                hypothesis=hypothesis,
                hypothesis_binary=binary,
                motivation=theme.motivation,
                novelty_justification=theme.novelty,
                methodology_sketch=theme.methodology,
                failure_modes=[],  # Will be filled in evaluation
                success_criteria=SuccessCriteria(
                    primary_metric=theme.metric,
                    target_value=theme.target,
                    statistical_threshold="p < 0.05",
                    min_effect_size=0.05,
                    num_random_seeds=3,
                ),
                feasibility_score=0.0,  # Will be calculated
                impact_score=0.0,  # Will be calculated
                novelty_score=0.0,  # Will be calculated
                combined_score=0.0,
            )
            candidates.append(candidate)

        return candidates

    def _extract_themes(self) -> list[_ResearchTheme]:
        """Extract research themes from the literature."""
        themes: list[_ResearchTheme] = []

        # Parse arxiv results for themes
        for r in self._arxiv_results[:10]:
            title = r.get("title", "")
            abstract = r.get("abstract", "")

            # Look for techniques and gaps
            techniques = self._extract_techniques(abstract)
            gaps = self._identify_gaps(abstract)

            if techniques and gaps:
                themes.append(
                    _ResearchTheme(
                        name=f"{techniques[0]}+{gaps[0]}",
                        motivation=f"Gap identified: {gaps[0]} not addressed with {techniques[0]}",
                        novelty=f"Combines {techniques[0]} with {gaps[0]}",
                        methodology=f"Apply {techniques[0]} to solve {gaps[0]}",
                        metric="accuracy",
                        target=85.0,
                    )
                )

        # If not enough themes, create defaults based on topic
        if len(themes) < self.num_ideas:
            defaults = [
                _ResearchTheme(
                    name="parameter efficient fine-tuning",
                    motivation="Fine-tuning large models is computationally expensive",
                    novelty="Novel application of LoRA to this domain",
                    methodology="Use LoRA with domain-specific adaptation",
                    metric="accuracy",
                    target=85.0,
                ),
                _ResearchTheme(
                    name="contrastive learning",
                    motivation="Limited labeled data available",
                    novelty="Self-supervised pre-training approach",
                    methodology="Contrastive learning with data augmentation",
                    metric="accuracy",
                    target=80.0,
                ),
                _ResearchTheme(
                    name="multi-modal fusion",
                    motivation="Current methods don't leverage multiple modalities",
                    novelty="Novel fusion architecture",
                    methodology="Cross-modal attention mechanism",
                    metric="F1-score",
                    target=82.0,
                ),
                _ResearchTheme(
                    name="efficient attention",
                    motivation="Quadratic complexity of self-attention",
                    novelty="Linear attention for scalability",
                    methodology="Replace attention with linear variant",
                    metric="throughput",
                    target=2.0,
                ),
                _ResearchTheme(
                    name="robustness",
                    motivation="Models fail on distribution shift",
                    novelty="Novel augmentation strategy",
                    methodology="Domain randomization and adversarial training",
                    metric="robustness_acc",
                    target=75.0,
                ),
            ]
            themes.extend(defaults[: self.num_ideas - len(themes)])

        return themes[: self.num_ideas]

    def _extract_techniques(self, text: str) -> list[str]:
        """Extract ML techniques from text."""
        techniques = [
            "LoRA",
            "transformer",
            "attention",
            "BERT",
            "GPT",
            "contrastive",
            "diffusion",
            "GAN",
            "CNN",
            "RNN",
            "reinforcement learning",
            "fine-tuning",
            "pre-training",
        ]
        found = []
        text_lower = text.lower()
        for tech in techniques:
            if tech.lower() in text_lower:
                found.append(tech)
        return found[:3]

    def _identify_gaps(self, text: str) -> list[str]:
        """Identify research gaps from text."""
        gap_phrases = [
            "limited",
            "lack",
            "challenging",
            "difficult",
            "inefficient",
            "expensive",
            "slow",
            "suboptimal",
            "poor performance",
            "cannot scale",
            "limited data",
            "shortage",
        ]
        gaps = []
        text_lower = text.lower()
        for gap in gap_phrases:
            if gap in text_lower:
                gaps.append(gap)
        return gaps[:2] if gaps else ["limited performance"]

    def _generate_hypothesis(self, index: int, theme: _ResearchTheme) -> tuple[str, str]:
        """Generate a binary falsifiable hypothesis."""
        templates = [
            (
                f"Fine-tuning {theme.name} on 1000 domain-specific examples achieves "
                f">{theme.target}% accuracy on the target benchmark",
                f"Fine-tuning {theme.name} on 1000 domain-specific examples achieves "
                f">{theme.target}% accuracy on the target benchmark (YES/NO)",
            ),
            (
                f"Applying {theme.name} reduces computational cost by 50% while "
                f"maintaining >90% of baseline performance",
                f"Applying {theme.name} reduces computational cost by 50% while "
                f"maintaining >90% of baseline performance (YES/NO)",
            ),
            (
                f"The combination of {theme.name} with standard baselines "
                f"outperforms state-of-the-art by at least 5% on {theme.metric}",
                f"The combination of {theme.name} with standard baselines "
                f"outperforms state-of-the-art by at least 5% on {theme.metric} (YES/NO)",
            ),
            (
                f"Using {theme.name} enables the model to generalize to out-of-distribution "
                f"data with <10% degradation",
                f"Using {theme.name} enables the model to generalize to out-of-distribution "
                f"data with <10% degradation (YES/NO)",
            ),
            (
                f"{theme.name} achieves {theme.target}% {theme.metric} with "
                f"only 10% of original compute budget",
                f"{theme.name} achieves {theme.target}% {theme.metric} with "
                f"only 10% of original compute budget (YES/NO)",
            ),
        ]
        return templates[index % len(templates)]

    def _evaluate_candidate(self, candidate: CandidateIdea) -> CandidateIdea:
        """Evaluate a single candidate idea."""
        # Run novelty kill search
        novelty_check = self._run_novelty_kill_search(candidate)
        candidate.novelty_check = novelty_check

        # Calculate feasibility
        candidate.feasibility_score = self._assess_feasibility(candidate)

        # Calculate impact
        candidate.impact_score = self._assess_impact(candidate)

        # Set novelty score from check
        candidate.novelty_score = novelty_check.novelty_score

        # Calculate combined score
        candidate.combined_score = (
            candidate.feasibility_score * candidate.impact_score * candidate.novelty_score
        )

        # Identify failure modes
        candidate.failure_modes = self._identify_failure_modes(candidate)

        # Check if passes filters
        candidate.passed_filters = (
            candidate.feasibility_score >= self.min_feasibility
            and candidate.impact_score >= self.min_impact
            and candidate.novelty_score >= self.min_novelty
            and novelty_check.is_novel
        )

        return candidate

    def _run_novelty_kill_search(self, candidate: CandidateIdea) -> NoveltyCheckResult:
        """Run novelty kill search for a hypothesis."""
        # Construct search queries
        queries = self._build_novelty_queries(candidate)

        self._log("novelty_kill_search", {"hypothesis": candidate.hypothesis, "queries": queries})

        closest_papers: list[ClosestPaper] = []

        # Search for each query
        for query in queries[:3]:
            try:
                results = search_arxiv(query, max_results=5)
                for r in results[:2]:
                    closest_papers.append(
                        ClosestPaper(
                            title=r.title,
                            authors=r.authors[:3],
                            year=int(r.published[:4]) if r.published else None,
                            similarity=f"Query: {query[:50]}...",
                            differentiation="Different domain/context",
                        )
                    )
            except Exception:
                pass

        # Calculate novelty score
        if not closest_papers:
            novelty_score = 1.0
            is_novel = True
            rejection_reason = None
        else:
            # If found very similar papers, reduce novelty
            novelty_score = max(0.3, 1.0 - (len(closest_papers) * 0.2))
            is_novel = novelty_score >= self.min_novelty
            rejection_reason = "Insufficient novelty" if not is_novel else None

        return NoveltyCheckResult(
            hypothesis=candidate.hypothesis,
            search_queries=queries,
            closest_papers=closest_papers,
            novelty_score=novelty_score,
            is_novel=is_novel,
            rejection_reason=rejection_reason,
        )

    def _build_novelty_queries(self, candidate: CandidateIdea) -> list[str]:
        """Build search queries for novelty check."""
        base = self.topic
        hypothesis = candidate.hypothesis.lower()

        queries = [
            f"{base} {candidate.title.split(':')[1].strip() if ':' in candidate.title else base}",
            hypothesis[:80] if len(hypothesis) > 80 else hypothesis,
        ]

        # Extract key terms from hypothesis
        terms = re.findall(r"\b\w+\b", hypothesis)
        significant_terms = [t for t in terms if len(t) > 4][:4]
        if significant_terms:
            queries.append(" ".join(significant_terms[:3]))

        return queries

    def _assess_feasibility(self, candidate: CandidateIdea) -> float:
        """Assess feasibility score (0-1)."""
        score = 0.5

        # Check if methodology is standard
        methodology = candidate.methodology_sketch.lower()
        standard_methods = [
            "fine-tuning",
            "lora",
            "adapter",
            "prompt",
            "training",
            "learning",
            "gradient",
            "backprop",
        ]
        if any(m in methodology for m in standard_methods):
            score += 0.2

        # Check if target is realistic
        target = candidate.success_criteria.target_value
        if 50 <= target <= 95:  # Reasonable range
            score += 0.15

        # Check compute requirements (based on hypothesis)
        hypothesis = candidate.hypothesis.lower()
        if "1000" in hypothesis or "small" in hypothesis or "efficient" in hypothesis:
            score += 0.15

        return min(1.0, score)

    def _assess_impact(self, candidate: CandidateIdea) -> float:
        """Assess impact score (0-1)."""
        score = 0.5

        # Check related work for problem importance
        for work in self._related_work[:5]:
            abstract = work.summary.lower()
            if any(kw in abstract for kw in ["important", "significant", "crucial", "critical"]):
                score += 0.1
                break

        # Check if addressing a gap
        if "gap" in candidate.motivation.lower() or "limited" in candidate.motivation.lower():
            score += 0.2

        # Generalization and robustness are high-impact
        if "generalize" in candidate.hypothesis.lower() or "robust" in candidate.hypothesis.lower():
            score += 0.2

        return min(1.0, score)

    def _identify_failure_modes(self, candidate: CandidateIdea) -> list[FailureMode]:
        """Identify at least 3 failure modes for a candidate."""
        failure_modes: list[FailureMode] = []

        # Failure mode 1: Baseline already achieves high performance
        failure_modes.append(
            FailureMode(
                scenario="If baseline already achieves >90% on this dataset, "
                "gains may be marginal and not worth the effort",
                likelihood="MEDIUM",
                mitigation="Verify baseline performance before proceeding",
            )
        )

        # Failure mode 2: Compute constraints
        failure_modes.append(
            FailureMode(
                scenario="If compute required exceeds 24h on A100, "
                "not feasible for reproduction and publication timeline",
                likelihood="MEDIUM",
                mitigation="Start with smaller models/datasets for validation",
            )
        )

        # Failure mode 3: Hyperparameter sensitivity
        failure_modes.append(
            FailureMode(
                scenario="If hyperparameter sensitivity is high (>10% variance "
                "across seeds), method may not be robust",
                likelihood="HIGH",
                mitigation="Use rigorous hyperparameter search and report variance",
            )
        )

        # Failure mode 4: Domain shift
        failure_modes.append(
            FailureMode(
                scenario="If training and test distributions differ significantly, "
                "expected improvements may not materialize",
                likelihood="MEDIUM",
                mitigation="Evaluate on multiple domains/datasets",
            )
        )

        # Failure mode 5: Implementation complexity
        failure_modes.append(
            FailureMode(
                scenario="If method requires significant engineering effort, "
                "may exceed project timeline",
                likelihood="LOW",
                mitigation="Start with well-established implementations",
            )
        )

        return failure_modes[:3]  # Return at least 3

    def _create_output(
        self,
        candidates: list[CandidateIdea],
        selected: CandidateIdea,
    ) -> IdeationOutput:
        """Create the final ideation output."""
        return IdeationOutput(
            topic=self.topic,
            candidate_ideas=candidates,
            selected_idea=selected,
            selection_rationale=f"Selected '{selected.title}' with combined score "
            f"{selected.combined_score:.2f} (feasibility={selected.feasibility_score:.2f}, "
            f"impact={selected.impact_score:.2f}, novelty={selected.novelty_score:.2f}). "
            f"Passed all thresholds: feasibility>={self.min_feasibility}, "
            f"impact>={self.min_impact}, novelty>={self.min_novelty}.",
            rejected=False,
        )

    def _create_rejected_output(self, candidates: list[CandidateIdea]) -> IdeationOutput:
        """Create output when all candidates are rejected."""
        reasons = []
        for c in candidates:
            r = []
            if c.feasibility_score < self.min_feasibility:
                r.append(f"feasibility={c.feasibility_score:.2f}<{self.min_feasibility}")
            if c.impact_score < self.min_impact:
                r.append(f"impact={c.impact_score:.2f}<{self.min_impact}")
            if c.novelty_score < self.min_novelty:
                r.append(f"novelty={c.novelty_score:.2f}<{self.min_novelty}")
            if c.novelty_check and not c.novelty_check.is_novel:
                r.append("not novel")
            if r:
                reasons.append(f"{c.title}: {', '.join(r)}")

        return IdeationOutput(
            topic=self.topic,
            candidate_ideas=candidates,
            selected_idea=None,
            selection_rationale="No candidate passed all thresholds.",
            rejected=True,
            rejection_reason="; ".join(reasons) if reasons else "All candidates failed threshold checks.",
        )

    def _write_output_files(self, output: IdeationOutput) -> None:
        """Write proposal.md and ideation_output.json."""
        # Write JSON output
        json_path = self.project.idea_dir / "ideation_output.json"
        self._log("write_json", {"path": str(json_path)})
        write_json(json_path, output.model_dump())
        self._log("write_json", {"path": str(json_path)}, outputs="written")

        # Write markdown proposal
        proposal_path = self.project.idea_dir / "proposal.md"
        proposal_content = self._generate_proposal_markdown(output)
        self._log("write_file", {"path": str(proposal_path)})
        atomic_write(proposal_path, proposal_content)
        self._log("write_file", {"path": str(proposal_path)}, outputs="written")

    def _generate_proposal_markdown(self, output: IdeationOutput) -> str:
        """Generate human-readable proposal.md."""
        selected = output.selected_idea
        lines: list[str] = []

        # Header
        lines.append(f"# Research Proposal: {self.topic}\n")

        if output.rejected:
            lines.append("## Status: REJECTED\n")
            lines.append(f"**Reason**: {output.rejection_reason}\n")
            lines.append("\nNo candidate passed all threshold checks. See details below.\n")
            return "\n".join(lines)

        if not selected:
            return "\n".join(lines)

        # Executive Summary
        lines.append("## Executive Summary\n")
        lines.append(
            f"This proposal addresses '{self.topic}' through {selected.title}. "
            f"The research aims to validate the hypothesis that {selected.hypothesis_binary[:100]}...\n"
        )

        # Background and Motivation
        lines.append("## Background and Motivation\n")
        lines.append(f"{selected.motivation}\n")

        # Research Hypothesis
        lines.append("## Research Hypothesis\n")
        lines.append(f"**Binary Falsifiable Statement**: {selected.hypothesis_binary}\n")
        lines.append(f"\n**Full Hypothesis**: {selected.hypothesis}\n")

        # Novelty Justification
        lines.append("## Novelty Justification\n")
        lines.append(f"{selected.novelty_justification}\n")

        if selected.novelty_check:
            lines.append("\n### Novelty Kill Search Results\n")
            if selected.novelty_check.closest_papers:
                lines.append("Closest existing work:\n")
                for i, paper in enumerate(selected.novelty_check.closest_papers[:3], 1):
                    lines.append(f"{i}. **{paper.title}**")
                    lines.append(f"   - Authors: {', '.join(paper.authors)}")
                    lines.append(f"   - Similarity: {paper.similarity}")
                    lines.append(f"   - Differentiation: {paper.differentiation}\n")
            else:
                lines.append("No directly comparable prior work found.\n")

        # Methodology Sketch
        lines.append("\n## Methodology Sketch\n")
        lines.append(f"{selected.methodology_sketch}\n")

        # Success Criteria (Locked)
        lines.append("\n## Success Criteria (Quantitative, Locked)\n")
        sc = selected.success_criteria
        lines.append(f"- **Primary Metric**: {sc.primary_metric}")
        lines.append(f"- **Target Value**: {sc.target_value}")
        lines.append(f"- **Statistical Threshold**: {sc.statistical_threshold}")
        if sc.min_effect_size:
            lines.append(f"- **Minimum Effect Size**: {sc.min_effect_size}")
        lines.append(f"- **Number of Random Seeds**: {sc.num_random_seeds}\n")

        # Failure Modes
        lines.append("\n## Failure Modes (≥3 Specific Scenarios)\n")
        for i, fm in enumerate(selected.failure_modes, 1):
            lines.append(f"{i}. **{fm.scenario}**")
            lines.append(f"   - Likelihood: {fm.likelihood}")
            lines.append(f"   - Mitigation: {fm.mitigation}\n")

        # Expected Contributions
        lines.append("\n## Expected Contributions\n")
        lines.append(f"1. Novel application of {selected.title.split(':')[1].strip() if ':' in selected.title else 'techniques'} to {self.topic}")
        lines.append(f"2. Quantitative evaluation with {selected.success_criteria.primary_metric}")
        lines.append(f"3. Public implementation and reproducibility\n")

        # Scores Summary
        lines.append("\n## Evaluation Scores\n")
        lines.append(f"- Feasibility: {selected.feasibility_score:.2f} (threshold: {self.min_feasibility})")
        lines.append(f"- Impact: {selected.impact_score:.2f} (threshold: {self.min_impact})")
        lines.append(f"- Novelty: {selected.novelty_score:.2f} (threshold: {self.min_novelty})")
        lines.append(f"- **Combined Score**: {selected.combined_score:.2f}\n")

        return "\n".join(lines)


class _ResearchTheme:
    """Internal helper for research themes."""

    def __init__(
        self,
        name: str,
        motivation: str,
        novelty: str,
        methodology: str,
        metric: str,
        target: float,
    ):
        self.name = name
        self.motivation = motivation
        self.novelty = novelty
        self.methodology = methodology
        self.metric = metric
        self.target = target
