# Project Checkpoints

## Refactor

1. Inspect the target notebook folder and any existing `src` package for overlapping logic.
2. Move reusable code into `src/` and keep notebook-side `.py` files as thin CLI wrappers.
3. Preserve existing output filenames, saved artifact names, and default parameters unless the user asks for behavior changes.
4. Route shared datalist and data-file access through `src/paths.py` instead of hard-coded `../data/...` paths.
5. Keep plotting-heavy or one-off exploratory code out of `src` unless it clearly becomes reusable.

## Validation

1. Run `python -m compileall` on each changed `src` package and notebook script directory.
2. Run `python <wrapper>.py --help` for every changed CLI runner.
3. Run a lightweight import or no-op smoke test for new reusable functions.
4. Avoid full dataset jobs unless the user explicitly wants them, especially when outputs go to `/nrs/...`.
5. Review `git diff --stat` and `git status --short --untracked-files=all` before commit.

## Docs

1. Update `README.md` only when the visible project structure or supported scripts changed.
2. Keep README entries high-signal: scripts, modules, and dataset indexes.
3. Do not document bulky local data assets file-by-file when they belong to the separate data repo.

## Git

1. Stage only the files that belong to the requested task.
2. Commit only after validation passes or after explicitly stating what could not be validated.
3. Push only when the user asks.
