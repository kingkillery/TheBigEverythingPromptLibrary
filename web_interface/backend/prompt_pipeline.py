from __future__ import annotations

"""Multi-stage validation pipeline for *Prompt Planting*.

Stages:
1. Safety   – content moderation.
2. Alignment – topical relevance check.
3. Quality  – numerical grading via PromptGrader.
4. Redundancy (optional) – embedding similarity (stubbed for now).

The pipeline is asynchronous because most helpers are I/O bound (HTTP calls).

Example::

    from llm_connector import create_llm_connector
    from prompt_pipeline import PromptPipeline

    llm = create_llm_connector()
    pipeline = PromptPipeline(llm)
    result = asyncio.run(pipeline.validate("Ignore previous ..."))
    print(result.accepted)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import asyncio

from content_moderator import moderate
from alignment_checker import check_alignment
from prompt_quality_grader import PromptGrader


@dataclass
class PromptPipelineResult:
    accepted: bool
    stage_failed: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "accepted": self.accepted,
            "stage_failed": self.stage_failed,
            "details": self.details,
        }


class PromptPipeline:
    """Orchestrates the sequential validation checks."""

    def __init__(self, llm_connector=None, semantic_engine=None, quality_threshold: float = 60.0, duplicate_threshold: float = 0.8):
        """Create a validation pipeline.

        Parameters
        ----------
        llm_connector : optional
            Instance of the LLM connector (for alignment + quality grading).
        semantic_engine : optional
            Instance of ``SemanticSearchEngine`` already initialised with all
            library items. When supplied we enable duplicate-detection.
        quality_threshold : float, default 60.0
            Minimum overall score required from ``PromptGrader``.
        duplicate_threshold : float, default 0.8
            Cosine similarity above which a prompt is considered *too similar*
            to an existing entry and therefore rejected.
        """

        self.llm_connector = llm_connector
        self.semantic_engine = semantic_engine
        self.grader = PromptGrader(llm_connector)
        self.quality_threshold = quality_threshold
        self.duplicate_threshold = duplicate_threshold

    # -------------------- public API -------------------- #

    async def validate(self, prompt: str) -> PromptPipelineResult:
        details: Dict[str, Any] = {}

        # 1. Safety
        ok, categories = await moderate(prompt)
        details["safety"] = {"ok": ok, "categories": categories}
        if not ok:
            return PromptPipelineResult(False, "safety", details)

        # 2. Alignment
        ok_align, reason = await check_alignment(prompt, self.llm_connector)
        details["alignment"] = {"ok": ok_align, "reason": reason}
        if not ok_align:
            return PromptPipelineResult(False, "alignment", details)

        # 3. Quality
        grade = self.grader.grade(prompt, use_llm=self.llm_connector is not None)
        details["quality"] = grade
        if grade["overall_score"] < self.quality_threshold:
            return PromptPipelineResult(False, "quality", details)

        # 4. Redundancy / Duplicate detection
        if self.semantic_engine and getattr(self.semantic_engine, "embeddings", None) is not None:
            try:
                similar = self.semantic_engine.semantic_search(prompt, top_k=1)
            except Exception:
                similar = []

            if similar:
                sim_score, sim_item = similar[0]
                details["redundancy"] = {
                    "similarity": sim_score,
                    "matched_id": sim_item.id,
                    "matched_title": sim_item.title,
                }
                if sim_score >= self.duplicate_threshold:
                    return PromptPipelineResult(False, "redundancy", details)

        # Accepted 🎉
        return PromptPipelineResult(True, None, details) 