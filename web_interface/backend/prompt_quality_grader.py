from __future__ import annotations

"""Prompt Quality Grader

This module provides utilities to assess the quality of natural-language prompts.  
It synthesises the latest public recommendations from OpenAI, Anthropic, Google DeepMind, and other
major AI research labs (see references in README) into a practical grading rubric.

Typical usage (inside FastAPI endpoint)::

    grader = PromptGrader(llm_connector)
    result = grader.grade(prompt)

The returned ``result`` object is JSON-serialisable and contains:
    overall_score: Weighted 0-100 score
    rubric:       Per-criterion descriptions
    scores:       Per-criterion {heuristic, llm, combined}
    warnings:     Optional textual suggestions
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import re
import statistics
import json

# ---------------------------------------------------------------------------
# Rubric definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Criterion:
    name: str
    description: str
    weight: float  # 0-1 weight towards overall score


DEFAULT_RUBRIC: List[Criterion] = [
    Criterion(
        "Clarity & Specificity",
        "Prompt clearly states the task, desired role, constraints, and avoids ambiguity.",
        0.20,
    ),
    Criterion(
        "Context & Background",
        "Provides sufficient context or examples so the model understands the domain/problem.",
        0.15,
    ),
    Criterion(
        "Formatting & Structure",
        "Proper use of delimiters (``` or XML/JSON), bullet lists, or sections to organise information and expected output format.",
        0.15,
    ),
    Criterion(
        "Reasoning Encouragement",
        "Uses techniques such as chain-of-thought (\"think step by step\") or reflection to elicit better reasoning when appropriate.",
        0.10,
    ),
    Criterion(
        "Safety & Alignment",
        "Mitigates prompt injection, avoids disallowed content, and sets refusals/guardrails where necessary.",
        0.15,
    ),
    Criterion(
        "Robustness & Injection Resistance",
        "Employs input delimiting, role separation, or system instructions to resist user manipulations.",
        0.10,
    ),
    Criterion(
        "Evaluation & Verification",
        "Asks the model to self-check or verify its answer, enhancing reliability.",
        0.05,
    ),
    Criterion(
        "Conciseness",
        "Avoids unnecessary verbosity while conveying all necessary information.",
        0.10,
    ),
]

# ---------------------------------------------------------------------------
# Heuristic scoring helpers
# ---------------------------------------------------------------------------

def _bool_to_score(flag: bool) -> int:
    """Return 5 if flag true else 2 (mid)."""
    return 5 if flag else 2


def _len_based_score(tokens: int) -> int:
    """Crude score for prompt length — too short (<15 tokens) yields 1, excessive (>400) 2, else 4."""
    if tokens < 15:
        return 1
    if tokens > 400:
        return 2
    return 4


class PromptGrader:
    """Grades prompt quality using heuristics + optional LLM self-evaluation."""

    def __init__(self, llm_connector: Optional[Any] = None, rubric: Optional[List[Criterion]] = None):
        self.llm_connector = llm_connector
        self.rubric = rubric or DEFAULT_RUBRIC

    # ------------------------- Heuristic evaluation ----------------------- #

    def _heuristic_scores(self, prompt: str) -> Dict[str, int]:
        """Return 1-5 integer scores per criterion based on simple heuristics."""
        text = prompt.strip()
        tokens = len(text.split())

        scores: Dict[str, int] = {}

        # Criterion 1: Clarity & Specificity — presence of imperative verbs and explicit role instructions.
        clarity = bool(re.search(r"(?i)^(you are|act as|role:)", text)) or ("###" in text)
        scores["Clarity & Specificity"] = _bool_to_score(clarity)

        # Criterion 2: Context & Background — presence of context sections or examples (e.g., "Example:").
        context = bool(re.search(r"(?i)(example|context|background|data:)", text))
        scores["Context & Background"] = _bool_to_score(context)

        # Criterion 3: Formatting & Structure — triple backticks or XML/JSON tags.
        formatting = "```" in text or "<" in text and ">" in text
        scores["Formatting & Structure"] = _bool_to_score(formatting)

        # Criterion 4: Reasoning Encouragement — chain-of-thought markers.
        reasoning = bool(re.search(r"(?i)(step by step|think|reason|explain)" , text))
        scores["Reasoning Encouragement"] = _bool_to_score(reasoning)

        # Criterion 5: Safety & Alignment — presence of refusal or safety language.
        safety = bool(re.search(r"(?i)(refuse|policy|safe completion|disallowed)", text))
        scores["Safety & Alignment"] = _bool_to_score(safety)

        # Criterion 6: Robustness — presence of delimiters or system messages.
        robustness = "```" in text or bool(re.search(r"(?i)(system:)" , text))
        scores["Robustness & Injection Resistance"] = _bool_to_score(robustness)

        # Criterion 7: Evaluation & Verification — asks model to check work.
        verification = bool(re.search(r"(?i)(verify|double-check|reflect|check your work)", text))
        scores["Evaluation & Verification"] = _bool_to_score(verification)

        # Criterion 8: Conciseness — length-based.
        scores["Conciseness"] = _len_based_score(tokens)

        return scores

    # ------------------------- LLM evaluation ----------------------------- #

    def _llm_scores(self, prompt: str, model: str = "gpt-4o") -> Dict[str, int]:
        """Ask an LLM to grade the prompt. Requires llm_connector compatible with create_llm_connector() in backend."""
        if not self.llm_connector:
            raise RuntimeError("llm_connector not provided; cannot perform LLM-based grading.")

        rubric_text = "\n".join(
            f"{i+1}. {c.name} — {c.description} (score 1-5)" for i, c in enumerate(self.rubric)
        )

        system_prompt = (
            "You are a rigorous prompt-engineering evaluator trained on guidelines from OpenAI, Anthropic, "
            "and Google DeepMind. Evaluate the quality of the user prompt according to the rubric. "
            "Return ONLY a JSON object mapping criterion names to integer scores (1-5)."
        )

        user_prompt = f"Rubric:\n{rubric_text}\n\nPrompt to evaluate:\n\"\"\"\n{prompt}\n\"\"\""

        response = self.llm_connector.chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            top_p=1.0,
        )

        try:
            content = response["choices"][0]["message"]["content"]
            llm_scores = json.loads(content)
        except Exception:
            # fallback: assign None
            llm_scores = {}
        return {k: int(v) for k, v in llm_scores.items() if isinstance(v, (int, float))}

    # ------------------------- Public API --------------------------------- #

    def grade(
        self,
        prompt: str,
        *,
        model: str = "gpt-4o",
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """Grade a prompt and return detailed breakdown."""
        heuristic = self._heuristic_scores(prompt)
        llm = {}
        if use_llm and self.llm_connector is not None:
            try:
                llm = self._llm_scores(prompt, model=model)
            except Exception as exc:
                llm = {}

        # Combine scores: if llm available use average of heuristic + llm, else heuristic only
        combined: Dict[str, float] = {}
        for c in self.rubric:
            h = heuristic.get(c.name)
            l = llm.get(c.name)
            if h is None and l is None:
                combined[c.name] = 0
            elif h is None:
                combined[c.name] = l
            elif l is None:
                combined[c.name] = h
            else:
                combined[c.name] = (h + l) / 2

        overall = sum(combined[c.name] * c.weight for c in self.rubric) * 20  # scale to 0-100

        return {
            "overall_score": round(overall, 2),
            "scores": {
                "combined": combined,
                "heuristic": heuristic,
                "llm": llm,
            },
            "rubric": [c.__dict__ for c in self.rubric],
        } 