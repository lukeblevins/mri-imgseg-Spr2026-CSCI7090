# Workspace Agent Instructions

## Notebook compatibility: Google Colab and local

When editing notebooks in this project, preserve execution compatibility for both Google Colab and local environments.

- Keep notebooks runnable top-to-bottom in both Google Colab and a local workspace.
- Do not hardcode machine-specific absolute paths.
- Preserve dual-path discovery patterns (for example local data folders, `EEG_DATA_ROOT`, Colab `/content`, and mounted Google Drive paths).
- Avoid Colab-only commands or dependencies unless guarded by runtime checks.
- Reuse existing environment helpers and configuration variables when available instead of introducing one-off setup logic.
- When adding setup steps, ensure there is both a local-friendly and Colab-friendly path.
- If a change introduces a compatibility tradeoff, document the tradeoff and provide a fallback for the other environment.

## Notebook writing style and capitalization

When editing notebook markdown text in this project, keep writing natural and concise while preserving technical accuracy.

Scope: these writing-style rules apply to markdown cells only, not code cells.

- Prefer human, direct phrasing over template-heavy or overly formal wording.
- Use consistent capitalization in headings and section labels.
- Keep section purpose statements short and specific.
- Preserve existing notebook structure and intent while improving clarity.
