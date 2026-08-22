# Doorman Archive

Public archive of the Doorman project resources:

- `github/GR00T-VisualSim2Real/`: working tree of the `doorman` branch from NVlabs/GR00T-VisualSim2Real.
- `github/GR00T-VisualSim2Real.bundle`: complete Git bundle of the archived repository refs and history.
- `arxiv/2512.01061/`: arXiv v1 source archive and extracted TeX sources.

## Restore the archived Git repository

```bash
git clone github/GR00T-VisualSim2Real.bundle GR00T-VisualSim2Real
git -C GR00T-VisualSim2Real switch doorman
```

Original sources:

- https://github.com/NVlabs/GR00T-VisualSim2Real/tree/doorman
- https://arxiv.org/abs/2512.01061

## Project notes maintained outside the frozen sources

- `teacher_training/` — teacher-training recipe, staged-reset/reward notes, and reproduction evidence.
- `evaluation/` — official evaluation semantics, paper/code distinction, current checkpoint results, and a reproducible 4090 evaluation runbook.
- `status/` — current experiment and decision records.
- `phase2_training/` — phase-two training notes.

The archived `github/` and `arxiv/` trees are source snapshots. Operational notes and locally reproduced results belong in the documentation folders above, not inside those snapshots.
