# Project Checkpoints

## Refactor

1. Inspect the target notebook folder and any existing `src` package for overlapping logic.
2. Move reusable code into `src/` and keep notebook-side `.py` files as thin CLI wrappers.
3. Keep the package name aligned with the notebook area when it has already been standardized, for example `notebooks/baseline_dynamics -> src/baseline_dynamics`.
4. Preserve existing output filenames, saved artifact names, and default parameters unless the user asks for behavior changes.
5. Route shared datalist and data-file access through `src/paths.py` instead of hard-coded `../data/...` paths.
6. Keep plotting-heavy or one-off exploratory code out of `src` unless it clearly becomes reusable.

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

## Notebook-Only Backup

1. When the task only changes ignored `*.ipynb` notebooks, record a concise checkpoint instead of attempting an empty Git commit.
2. Capture the exact notebook paths, the user-facing analysis state, and any manual conventions that would matter on rerun, such as axis swaps, dropped fish, threshold choices, or renamed headers.
3. Distinguish clearly between tracked code changes and local-only notebook changes so the backup does not imply that Git contains the notebook state.
4. For notebook-generated CSVs or figures that are part of the working state, note whether they were rewritten directly from the notebook logic or touched manually afterward.
5. If a notebook export schema was renamed for interpretation, record both the semantic labels and the underlying threshold meaning.

## Git

1. Stage only the files that belong to the requested task.
2. Commit only after validation passes or after explicitly stating what could not be validated.
3. Push only when the user asks.
