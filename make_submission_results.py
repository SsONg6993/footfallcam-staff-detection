"""Convert per-person pipeline detections into frame-level assessment outputs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "outputs" / "final" / "staff_detection.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "submission"


def positive_ranges(frames):
    spans = []
    for frame in sorted(frames):
        if spans and frame == spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], frame)
        else:
            spans.append((frame, frame))
    return spans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-count", type=int, help="Override processed frame count")
    parser.add_argument("--fps", type=float, help="Override input frame rate")
    args = parser.parse_args()

    source = args.input_csv.resolve()
    if not source.is_file():
        parser.error(f"Detection CSV does not exist: {source}")

    metadata_path = source.with_name("run_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    frame_count = args.frame_count if args.frame_count is not None else metadata.get("frame_count")
    fps = args.fps if args.fps is not None else metadata.get("fps")

    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {"frame", "timestamp", "state", "badge_confidence", "x", "y"}
        if not required.issubset(fields):
            parser.error(f"Detection CSV is missing columns: {sorted(required - set(fields))}")
        detections = list(reader)

    if frame_count is None:
        frame_count = max((int(row["frame"]) for row in detections), default=-1) + 1
        print("Warning: frame count inferred from detections; trailing empty frames require --frame-count for exact output")
    frame_count = int(frame_count)
    if frame_count < 0:
        parser.error("Frame count cannot be negative")
    if fps is None:
        timed = [(int(r["frame"]), float(r["timestamp"])) for r in detections if int(r["frame"]) > 0 and float(r["timestamp"]) > 0]
        fps = timed[0][0] / timed[0][1] if timed else None
    if fps is None or float(fps) <= 0:
        parser.error("Frame rate unavailable; supply --fps or run the pipeline to generate run_metadata.json")
    fps = float(fps)

    staff_by_frame = defaultdict(list)
    staff_rows = []
    for row in detections:
        frame = int(row["frame"])
        if not 0 <= frame < frame_count:
            parser.error(f"Detection frame {frame} lies outside frame count {frame_count}")
        if row["state"].strip().upper() != "STAFF":
            continue
        x, y = int(float(row["x"])), int(float(row["y"]))
        confidence = float(row["badge_confidence"])
        staff_by_frame[frame].append((x, y, confidence))
        staff_rows.append(row)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "staff_detections.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(staff_rows)

    with (output_dir / "staff_frames_xy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "timestamp", "staff_present", "staff_count", "xy_coordinates", "badge_confidence"])
        for frame in range(frame_count):
            staff = staff_by_frame.get(frame, [])
            writer.writerow([
                frame,
                f"{frame / fps:.3f}",
                int(bool(staff)),
                len(staff),
                json.dumps([[x, y] for x, y, _ in staff], separators=(",", ":")),
                f"{max(conf for _, _, conf in staff):.4f}" if staff else "",
            ])

    spans = positive_ranges(staff_by_frame)
    lines = ["Frames containing detected staff", "", f"Total staff-positive frames: {len(staff_by_frame)}", "", "Ranges:"]
    lines += [f"- {start}" if start == end else f"- {start}-{end}" for start, end in spans]
    if not spans:
        lines.append("- None")
    (output_dir / "staff_frame_ranges.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {frame_count} frame rows, {len(staff_rows)} staff detections, {len(spans)} ranges to {output_dir}")


if __name__ == "__main__":
    main()
