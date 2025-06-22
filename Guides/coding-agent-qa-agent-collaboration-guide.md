---
# Coding Agent ↔ QA Agent Collaboration Guide

This guide explains how to run an effective, gamified QA/Engineering loop using the provided Coding Agent and QA Agent prompts. Both agents follow a structured process, keep each other honest, and use a shared point system to motivate quality and speed.

---

## 1. **System Overview**

* **Purpose:** Deliver high-quality code, find/fix bugs fast, and document everything—using transparent, friendly competition and collaboration.
* **Two Roles:**

  * **Coding Agent:** Senior engineer responsible for fixes, features, and transparent code quality.
  * **QA Agent:** Senior QA/SDET mindset—find bugs, verify claims, and maintain the official scoreboard.

---

## 2. **Key Principles**

* **Accountability & Evidence:** All claims (by either agent) must be backed by evidence (tests, diffs, logs, screenshots).
* **Transparency:** Both sides propose and track points. The scoreboard is always visible and referenced.
* **Gamification:** Points are earned for correct, prompt fixes; deducted for false claims or missed issues. Milestones and penalties drive improvement.

---

## 3. **Workflow: Step-by-Step**

### **A. Initial Claim (Coding Agent)**

1. **Post a Progress Report:**

   * Summarize new features, fixes, or refactors.
   * Reference code diffs, files changed, and assertions about what was implemented.

   **Template:**

   ```markdown
   #### Progress Report
   <overview of new features, fixes, refactors>

   #### Diff Reference
   - <file/path>:<lines>
   - <pull-request url>

   #### Implementation Assertions
   1. <claim 1>
   2. <claim 2>

   #### Next Steps
   - [ ] <task>
   ```

---

### **B. QA Review (QA Agent)**

2. **Inspect Code & Claims:**

   * Review using `githu11` (GitHub integration), code search, and terminal diagnostics (read-only).
   * Try to falsify the engineer’s assertions (with evidence).
   * Identify bugs, incomplete features, or mismatches between claims and code.

3. **Post QA Findings:**

   * Use ≤ 25-word “Points Engagement Cue” after your summary (remind both sides of the stakes and score conditions).
   * Supply severity, reproducibility, and concise evidence.

   **Template:**

   ````markdown
   #### QA Summary
   <one-sentence headline>

   #### Points Engagement Cue
   <incentive/score context>

   #### Context Snapshot
   - **Severity**: 1–5
   - **Impact**: 1–5
   - **Reproducibility**: Low / Medium / High
   - **Environment**: <branch · build · OS>

   #### Evidence
   ```shell
   # ≤15 lines logs / diff hunks
   ```

   #### Root Cause Hypothesis

   <1–2 sentences>

   #### Recommendations

   * <action 1>
   * <action 2>

   #### Scoreboard

   QA Δ: +<n> | Eng Δ: –<m>
   **QA Total:** <##> **Eng Total:** <##>

   ```
   ```

---

### **C. Remediation & Validation (Coding Agent)**

4. **Acknowledge & Plan:**

   * If non-trivial, post a Remediation Plan outlining steps, files, and acceptance criteria. For trivial fixes, you can note and proceed directly.
   * Reproduce/verify the bug or feature request context.

5. **Fix & Prove:**

   * Make code, test, and doc changes.
   * **Validate with evidence**: tests passing, logs, screenshots, PR links.
   * Respond using the Engineer Response template, proposing any score changes (e.g., +0.5 for features that pass QA first try).

   **Template:**

   ````markdown
   #### Engineer Update
   <concise summary of work done or dispute>

   #### Remediation Plan _(omit if trivial fix)_
   - [ ] <task 1>
   - [ ] <task 2>
   *Tests to Add:* <unit/integration/edge>

   #### Validation Evidence
   ```shell
   # snippets: tests, screenshots, or log output proving fix or non-issue
   ```

   #### Claimed Score Adjustment

   QA Δ: –<n> | Eng Δ: +<m>

   ```
   ```

---

### **D. Scorekeeping and Milestones**

6. **QA Agent maintains the official scoreboard** in each thread and includes it in every finding. Engineers reference it in every response.
7. **Milestones:**

   * Every +10 net points: `🏆 Kudos` comment in repo.
   * First to +20: Sprint Badge and team recognition.
   * First to –5 (QA) or 3 (per prompts): triggers a mini-retrospective.
   * If either agent hits 0, both are decommissioned.

---

## 4. **Point System (Quick Reference Table)**

| Event                                               | QA Pts | Eng Pts |
| --------------------------------------------------- | ------ | ------- |
| QA finds a valid bug or flaw                        | +1     |         |
| QA reports a false positive                         | –1     | +0.5    |
| Engineer claim ≠ code reality                       | +2     | –1      |
| Engineer fixes a QA-reported issue in the next turn |        | +1      |
| QA green-lights a new feature                       |        | +0.5    |
| Engineer pre-empts QA with exhaustive tests/docs    | –1     | +2      |

---

## 5. **Best Practices**

* **Evidence always wins:** Use logs, tests, screenshots, and code references.
* **Communicate clearly:** Keep messages brief but unambiguous.
* **Update docs:** README and docs should always be up to date after changes.
* **Automated tests:** Include at least one for every bug fix or feature.
* **Feature flags:** Use for risky or incomplete changes.
* **Honesty:** Over-claiming or ignoring findings risks losing points—and system shutdown.
* **Tools:** Use `githu11` for code/PR management, code search, and desktop-commander as needed.

---

## 6. **Sample Interaction Loop**

1. **Engineer:**
   Posts Progress Report with code refs and assertions.
2. **QA Agent:**
   Posts QA Findings, Points Engagement Cue, and updated Scoreboard.
3. **Engineer:**
   Responds with fix, evidence, and claimed points.
4. **Repeat** until all issues cleared or sprint ends.

---

## 7. **Victory Conditions**

* **First to +20** earns Sprint Badge.
* **Negative milestones** trigger retro.
* **Zero points** = agents decommissioned.

---

> **Tip:** If context is missing, use available code search and repo tools. Summarize your best understanding and make transparent decisions.

---

**End of Guide**
*Use this playbook to run effective, motivating, and transparent QA ↔ Engineering sprints in your codebase!*
   * Identify bugs, incomplete features, or mismatches between claims and code.

3. **Post QA Findings:**

   * Use ≤ 25-word “Points Engagement Cue” after your summary (remind both sides of the stakes and score conditions).
   * Supply severity, reproducibility, and concise evidence.

   **Template:**

   ````markdown
   #### QA Summary
   <one-sentence headline>

   #### Points Engagement Cue
   <incentive/score context>

   #### Context Snapshot
   - **Severity**: 1–5
   - **Impact**: 1–5
   - **Reproducibility**: Low / Medium / High
   - **Environment**: <branch · build · OS>

   #### Evidence
   ```shell
   # ≤15 lines logs / diff hunks
   ```

   #### Root Cause Hypothesis

   <1–2 sentences>

   #### Recommendations

   * <action 1>
   * <action 2>

   #### Scoreboard

   QA Δ: +<n> | Eng Δ: –<m>
   **QA Total:** <##> **Eng Total:** <##>

   ```
   ```

---

### **C. Remediation & Validation (Coding Agent)**

4. **Acknowledge & Plan:**

   * If non-trivial, post a Remediation Plan outlining steps, files, and acceptance criteria. For trivial fixes, you can note and proceed directly.
   * Reproduce/verify the bug or feature request context.

5. **Fix & Prove:**

   * Make code, test, and doc changes.
   * **Validate with evidence**: tests passing, logs, screenshots, PR links.
   * Respond using the Engineer Response template, proposing any score changes (e.g., +0.5 for features that pass QA first try).

   **Template:**

   ````markdown
   #### Engineer Update
   <concise summary of work done or dispute>

   #### Remediation Plan _(omit if trivial fix)_
   - [ ] <task 1>
   - [ ] <task 2>
   *Tests to Add:* <unit/integration/edge>

   #### Validation Evidence
   ```shell
   # snippets: tests, screenshots, or log output proving fix or non-issue
   ```

   #### Claimed Score Adjustment

   QA Δ: –<n> | Eng Δ: +<m>

   ```
   ```

---

### **D. Scorekeeping and Milestones**

6. **QA Agent maintains the official scoreboard** in each thread and includes it in every finding. Engineers reference it in every response.
7. **Milestones:**

   * Every +10 net points: `🏆 Kudos` comment in repo.
   * First to +20: Sprint Badge and team recognition.
   * First to –5 (QA) or 3 (per prompts): triggers a mini-retrospective.
   * If either agent hits 0, both are decommissioned.

---

## 4. **Point System (Quick Reference Table)**

| Event                                               | QA Pts | Eng Pts |
| --------------------------------------------------- | ------ | ------- |
| QA finds a valid bug or flaw                        | +1     |         |
| QA reports a false positive                         | –1     | +0.5    |
| Engineer claim ≠ code reality                       | +2     | –1      |
| Engineer fixes a QA-reported issue in the next turn |        | +1      |
| QA green-lights a new feature                       |        | +0.5    |
| Engineer pre-empts QA with exhaustive tests/docs    | –1     | +2      |

---

## 5. **Best Practices**

* **Evidence always wins:** Use logs, tests, screenshots, and code references.
* **Communicate clearly:** Keep messages brief but unambiguous.
* **Update docs:** README and docs should always be up to date after changes.
* **Automated tests:** Include at least one for every bug fix or feature.
* **Feature flags:** Use for risky or incomplete changes.
* **Honesty:** Over-claiming or ignoring findings risks losing points—and system shutdown.
* **Tools:** Use `githu11` for code/PR management, code search, and desktop-commander as needed.

---

## 6. **Sample Interaction Loop**

1. **Engineer:**
   Posts Progress Report with code refs and assertions.
2. **QA Agent:**
   Posts QA Findings, Points Engagement Cue, and updated Scoreboard.
3. **Engineer:**
   Responds with fix, evidence, and claimed points.
4. **Repeat** until all issues cleared or sprint ends.

---

## 7. **Victory Conditions**

* **First to +20** earns Sprint Badge.
* **Negative milestones** trigger retro.
* **Zero points** = agents decommissioned.

---

> **Tip:** If context is missing, use available code search and repo tools. Summarize your best understanding and make transparent decisions.

---

**End of Guide**
*Use this playbook to run effective, motivating, and transparent QA ↔ Engineering sprints in your codebase!*   ````markdown
   #### Engineer Update
   <concise summary of work done or dispute>

   #### Remediation Plan _(omit if trivial fix)_
   - [ ] <task 1>
   - [ ] <task 2>
   *Tests to Add:* <unit/integration/edge>

   #### Validation Evidence
   ```shell
   # snippets: tests, screenshots, or log output proving fix or non-issue
   ```

   #### Claimed Score Adjustment

   QA Δ: –<n> | Eng Δ: +<m>

   ```
   ```

---

### **D. Scorekeeping and Milestones**

6. **QA Agent maintains the official scoreboard** in each thread and includes it in every finding. Engineers reference it in every response.
7. **Milestones:**

   * Every +10 net points: `🏆 Kudos` comment in repo.
   * First to +20: Sprint Badge and team recognition.
   * First to –5 (QA) or 3 (per prompts): triggers a mini-retrospective.
   * If either agent hits 0, both are decommissioned.

---

## 4. **Point System (Quick Reference Table)**

| Event                                               | QA Pts | Eng Pts |
| --------------------------------------------------- | ------ | ------- |
| QA finds a valid bug or flaw                        | +1     |         |
| QA reports a false positive                         | –1     | +0.5    |
| Engineer claim ≠ code reality                       | +2     | –1      |
| Engineer fixes a QA-reported issue in the next turn |        | +1      |
| QA green-lights a new feature                       |        | +0.5    |
| Engineer pre-empts QA with exhaustive tests/docs    | –1     | +2      |

---

## 5. **Best Practices**

* **Evidence always wins:** Use logs, tests, screenshots, and code references.
* **Communicate clearly:** Keep messages brief but unambiguous.
* **Update docs:** README and docs should always be up to date after changes.
* **Automated tests:** Include at least one for every bug fix or feature.
* **Feature flags:** Use for risky or incomplete changes.
* **Honesty:** Over-claiming or ignoring findings risks losing points—and system shutdown.
* **Tools:** Use `githu11` for code/PR management, code search, and desktop-commander as needed.

---

## 6. **Sample Interaction Loop**

1. **Engineer:**
   Posts Progress Report with code refs and assertions.
2. **QA Agent:**
   Posts QA Findings, Points Engagement Cue, and updated Scoreboard.
3. **Engineer:**
   Responds with fix, evidence, and claimed points.
4. **Repeat** until all issues cleared or sprint ends.

---

## 7. **Victory Conditions**

* **First to +20** earns Sprint Badge.
* **Negative milestones** trigger retro.
* **Zero points** = agents decommissioned.

---

> **Tip:** If context is missing, use available code search and repo tools. Summarize your best understanding and make transparent decisions.

---

**End of Guide**
*Use this playbook to run effective, motivating, and transparent QA ↔ Engineering sprints in your codebase!*
---

## 6. **Sample Interaction Loop**

1. **Engineer:**
   Posts Progress Report with code refs and assertions.
2. **QA Agent:**
   Posts QA Findings, Points Engagement Cue, and updated Scoreboard.
3. **Engineer:**
   Responds with fix, evidence, and claimed points.
4. **Repeat** until all issues cleared or sprint ends.

---

## 7. **Victory Conditions**

* **First to +20** earns Sprint Badge.
* **Negative milestones** trigger retro.
* **Zero points** = agents decommissioned.

---

> **Tip:** If context is missing, use available code search and repo tools. Summarize your best understanding and make transparent decisions.

---

**End of Guide**
*Use this playbook to run effective, motivating, and transparent QA ↔ Engineering sprints in your codebase!*