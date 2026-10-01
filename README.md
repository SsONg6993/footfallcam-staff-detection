# FootfallCam Staff Detection Assessment

## Overview

This project identifies frames containing staff in overhead CCTV video and reports an approximate image-space location for each detected staff member. A staff member is defined by a visible chest name tag / ID badge.

## Problem

The camera has a high, fisheye view. People may be seated, partly hidden, or directly below the camera. The badge is small and can disappear as a person turns or is occluded.

## Assumption

Only a visible detected badge establishes `STAFF` in the current frame. A person without current-frame badge evidence is reported as `PERSON`, even if that person was staff in an earlier frame.

## Solution Architecture

`Video frame → YOLO26s person detection → upper-body crop → YOLO26n badge detection → STAFF / PERSON → annotated video and CSV`

The pipeline tracks people for short-term ID and bounding-box continuity. Tracking does not assign or propagate staff identity.

## Models

- `models/person_best.pt`: fine-tuned YOLO26s person detector adapted to overhead / fisheye CCTV, including seated and partly occluded people.
- `models/badge_best.pt`: fine-tuned YOLO26n detector for the single `staff_tag` class, run on each person's upper-body crop.

The packaged files are copies of the active final weights. No model is trained at runtime.

## Staff Detection Logic

Badge detection on the current upper-body crop determines `STAFF`. This avoids transferring a staff label when people overlap or tracking IDs switch. Detection may therefore be intermittent when the badge is too small, turned away, or covered.

## XY Coordinate Definition

For person box `(x1, y1, x2, y2)`, the approximate location is its bottom-center: `x = (x1 + x2) / 2`, `y = y2`. Coordinates are pixels in the original video frame, with the origin at the top-left. They are not calibrated world coordinates. The pipeline's per-person `x,y` fields already implement this definition; the submission converter reuses them.

## Project Structure

```text
main.py                  CLI entry point
config.py                model paths and detection settings
pipeline.py              detection, tracking integration, video / CSV export
tracker.py               short-term person tracking
make_submission_results.py  frame-level assessment outputs
models/                  final packaged person and badge weights
training/                training scripts for reference only
docs/                    assessment PDF and cleanup report
outputs/                 generated results (ignored by Git)
```

## Installation

Use Python 3.10 or newer. From the project root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the assessment `sample.mp4` in the project root if running the default command. Video files are intentionally excluded from Git.

## How to Run

```bash
python main.py                         # default sample.mp4
python main.py --video new_test_video.mp4
python main.py --video new_test_video.mp4 --output-dir outputs/final
python make_submission_results.py
```

For a quick local check, `python main.py --video sample.mp4 --max-frames 5` processes five frames. Run the submission converter after a full run to produce complete results.

## Output Files

- `outputs/final/staff_detection.mp4`: annotated video.
- `outputs/final/staff_detection.csv`: one row per visible tracked person, including state, confidence and pixel `x,y`.
- `outputs/final/run_metadata.json`: processed frame count and source frame rate, including frames with no detected people.
- `outputs/submission/staff_detections.csv`: staff-only detection rows.
- `outputs/submission/staff_frames_xy.csv`: one row per processed frame, including `staff_present`, `staff_count`, JSON `xy_coordinates`, and maximum badge confidence for staff in that frame.
- `outputs/submission/staff_frame_ranges.txt`: consecutive staff-positive frame ranges.

If converting an older per-person CSV without `run_metadata.json`, pass `--frame-count` and `--fps` to `make_submission_results.py` for an exact frame-level export.

## Training / Fine-tuning

The supplied weights were produced by fine-tuning YOLO26s on manually reviewed overhead person annotations and YOLO26n on staff badge crops. `training/` holds the training scripts for reproducibility; running the interview demo does not run them. The latest annotation revisions are not claimed to have been incorporated into these weights.

## Limitations

Training data cover limited overhead CCTV conditions, so performance on unseen cameras may vary. Very small, blurred, turned-away or occluded badges may be missed. Tracking may temporarily lose IDs during overlaps. XY is an image-space approximation only.

## Interview Demo Instructions

From the project root, confirm both files in `models/` are present. Install dependencies, then run `python main.py --video new_test_video.mp4`. After processing, run `python make_submission_results.py`; show the annotated MP4, frame-level CSV and positive-frame ranges from `outputs/`. For a short check first, use `--max-frames 5`.
