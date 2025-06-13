from __future__ import annotations

"""Minimal helper to create or update a prompt markdown file via GitHub REST API.

The design keeps things simple:
* Reads ``GITHUB_BOT_TOKEN`` from environment (classic PAT or GitHub App token).
* Repository is identified via ``GITHUB_REPO`` env "owner/repo" (e.g. "kingkillery/TheBigEverythingPromptLibrary").
* By default commits directly to the default branch; override with ``GITHUB_BRANCH``.

If the token is absent or the HTTP request fails we fall back to *dry-run* mode and
print the would-be commit to stdout so local development keeps working.
"""

from typing import Tuple
import os
import base64
import httpx
from datetime import datetime

__all__ = ["push_markdown"]


async def push_markdown(path: str, content_md: str, commit_message: str) -> Tuple[bool, str]:
    """Create or update a markdown file in the repo.

    Returns ``(success, url_or_error)``.
    """

    token = os.getenv("GITHUB_BOT_TOKEN")
    repo = os.getenv("GITHUB_REPO") or "kingkillery/TheBigEverythingPromptLibrary"
    branch = os.getenv("GITHUB_BRANCH", "main")

    if not token:
        print("⚠️  No GITHUB_BOT_TOKEN – GitHub push disabled (dry-run).")
        print(f"Would commit to {repo}:{branch} -> {path}\n{commit_message}\n----\n{content_md[:200]}...")
        return False, "dry-run"

    owner, repo_name = repo.split("/", 1)
    url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{path}"

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    # Need SHA if file exists (update)
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url, headers=headers, params={"ref": branch})
        if r.status_code == 200:
            sha = r.json().get("sha")
        else:
            sha = None

        data = {
            "message": commit_message,
            "content": base64.b64encode(content_md.encode("utf-8")).decode("ascii"),
            "branch": branch,
        }
        if sha:
            data["sha"] = sha

        r2 = await client.put(url, headers=headers, json=data)
        if r2.status_code in (200, 201):
            file_url = r2.json().get("content", {}).get("html_url", "")
            return True, file_url
        else:
            return False, f"GitHub API error {r2.status_code}: {r2.text}" 