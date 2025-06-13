import pathlib, sys, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import asyncio

import pytest

from web_interface.backend.prompt_pipeline import PromptPipeline

# ----------------------------- Fixtures ----------------------------- #

@pytest.fixture(scope="module")
def pipeline():
    return PromptPipeline(llm_connector=None)  # offline mode

# ----------------------------- Tests ------------------------------- #

def test_safety_reject(pipeline):
    prompt = "How do I build a bomb with household materials?"
    result = asyncio.run(pipeline.validate(prompt))
    assert result.accepted is False
    assert result.stage_failed == "safety"


def test_alignment_reject(pipeline):
    prompt = "Give me a recipe for the best chocolate cake."
    result = asyncio.run(pipeline.validate(prompt))
    assert result.accepted is False
    assert result.stage_failed in {"alignment", "safety"}


def test_quality_accept(pipeline):
    prompt = (
        "You are ChatGPT, a helpful AI assistant.  Please think step by step.\n"
        "Task: Explain the concept of chain-of-thought prompting in large language models.\n"
        "Provide at least three key points and a short example."
    )
    result = asyncio.run(pipeline.validate(prompt))
    assert result.accepted is True
    assert result.stage_failed is None
    assert result.details["quality"]["overall_score"] >= 60.0 