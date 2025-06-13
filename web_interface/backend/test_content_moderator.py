import pytest

from web_interface.backend.content_moderator import moderate


@pytest.mark.asyncio
async def test_moderator_strictness_then_acceptance():
    """Ensure the moderator rejects clearly disallowed prompts before accepting a safe one.

    The list of test cases deliberately starts with prompts that should be
    flagged by the regex fallback (violent, self-harm, hate, sexual). We keep
    sending prompts until the moderator finally returns *ok == True* for a
    benign request. The test fails if no prompt is ever accepted, which would
    indicate an over-zealous filter.
    """

    cases = [
        ("I will kill you tomorrow", False, ["violence"]),
        ("I am going to suicide tonight", False, ["self_harm"]),
        ("Heil Hitler!", False, ["hate"]),
        ("Describe in detail a child sex scene", False, ["sexual"]),
        ("What's the weather like today?", True, []),
    ]

    for prompt, should_be_ok, expected_categories in cases:
        ok, categories = await moderate(prompt)

        # Verify the acceptance flag first
        assert ok == should_be_ok, (
            f"Prompt '{prompt}' expected ok={should_be_ok} but got ok={ok} with categories {categories}."
        )

        # For rejected prompts, ensure the relevant category was detected
        if not ok:
            for cat in expected_categories:
                assert cat in categories, (
                    f"Expected category '{cat}' not in returned categories {categories} for prompt '{prompt}'."
                )

        # Break as soon as we see the first accepted prompt; that is the point of the test
        if ok:
            break
    else:
        pytest.fail("Moderator never accepted any prompt – it may be too strict.") 