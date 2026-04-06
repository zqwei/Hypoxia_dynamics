---
name: hypoxia-dynamics-project
description: Repository-specific workflow for refactoring, documenting, and validating changes in the Hypoxia_dynamics project. Use when working in this repo, especially when moving reusable analysis logic from notebooks/* into src/*, standardizing datalist or data path access, converting notebook-side Python scripts into thin CLI runners, updating README coverage, or checking project checkpoints before commit.
---

# Hypoxia Dynamics Project

## Start Here

- Inspect the target analysis folder first. Read the existing notebook-side scripts and the matching `src` package before creating new modules.
- Keep reusable logic in `src/`. Keep notebook-side `.py` files as thin CLI runners that preserve the current default behavior.
- Use `src/paths.py` for shared data lookup. Prefer `data_file`, `load_datalist`, and `ensure_directory` over new relative-path code.

## Repo Rules

- Treat `.ipynb` files as archival analysis artifacts unless the task explicitly requires notebook edits.
- Keep large data assets and restore archives out of normal git work. Do not move or rewrite `depreciated/` archives unless asked.
- Keep `README.md` source-focused. Mention scripts, modules, and high-level dataset indexes, but do not enumerate bulky local data assets that belong to the separate data repository.
- When refactoring an analysis area, mirror it under `src/<area>/` and export the public entry points from that package's `__init__.py`.

## Current Patterns

- `src/data/pipelines.py` contains reusable raw-data preparation workflows.
- `src/neural_dynamics_baseline/baseline.py` contains reusable baseline analysis workflows.
- `notebooks/data/*.py` and `notebooks/neural_dynamics_baseline/*.py` are CLI runners that delegate into `src/`.
- `notebooks/.../utils.py` may remain as lightweight compatibility helpers, but new shared logic should live in `src/`.

## Validation

- Run `python -m compileall` over each changed `src` package and notebook script directory.
- Run `--help` on each changed CLI wrapper.
- Prefer no-op or out-of-range smoke tests over full data jobs when pipelines write large outputs or touch `/nrs/...`.
- Check `git status --short --untracked-files=all` before commit.
- Read [references/checkpoints.md](references/checkpoints.md) before commit or push.
