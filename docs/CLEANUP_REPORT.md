# Packaging and cleanup report

## Active submission path

`main.py` invokes `pipeline.py` and `tracker.py`. `config.py` selects the packaged weights. `make_submission_results.py` converts the per-person detection CSV into staff-only and frame-level assessment outputs. The detection thresholds, torso-crop geometry, badge classification rule, and tracking behavior were left unchanged.

## Files copied and moved

- Copied `runs/detect/runs/person/yolo26s_topdown/weights/best.pt` to `models/person_best.pt` (SHA-256 `03bdb80f5ae489c7560b4a0763858863c17d56a30e5e8749cbb2242e13952efe`).
- Copied `runs/detect/runs/badge/yolo26n_badge-4/weights/best.pt` to `models/badge_best.pt` (SHA-256 `788b07075cee1cdfd546f29ee84a7ad1a1558d09e16eb942b60b9b844ed9da65`).
- Moved `fine_tune_person_detector.py` to `training/fine_tune_person_clean.py`, `train_badge.py` to `training/train_badge.py`, and `test_badge.py` to `training/test_badge.py`.

## Kept locally, excluded from Git

- `optimized_final_pipeline.py`: experimental alternate implementation, not the active CLI pipeline.
- `custom_bytetrack.yaml`: unused legacy tracker configuration.
- `sample.mp4` and `synthetic_holdout_test.mp4`: assessment/local video assets; publishing rights have not been established.
- `yolo26n.pt` and `yolo26s.pt`: base weights, superseded for runtime by packaged final weights.
- `runs/`, `weights/`, `outputs/` generated results and annotation work, local caches and temporary files. `outputs/.gitkeep` is the only intended committed item under `outputs/`.

No legacy source, raw dataset, model weight, or video was deleted. The existing generated outputs remain available locally but are ignored by Git.

## Submission check

The repository was initialized locally, with no files staged or committed and no remote configured by this work. Check `git status` in the user's normal shell, then review before `git add .` and commit. A sandbox account may see Git's ownership warning; that is an environment-specific check rather than a repository defect.
