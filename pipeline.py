import csv
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import (
    BADGE_CONF,
    BADGE_IMGSZ,
    BADGE_MODEL,
    OUTPUT_DIR,
    PERSON_CONF,
    PERSON_IMGSZ,
    PERSON_MODEL,
    VIDEO_PATH,
)

from tracker import PersonTracker


class StaffDetectionPipeline:
    def __init__(self, video_path=VIDEO_PATH, output_dir=OUTPUT_DIR):
        project_root = Path(__file__).resolve().parent
        video = Path(video_path)
        self.video_path = video if video.is_absolute() or video.exists() else project_root / video
        output = Path(output_dir)
        self.output_dir = output if output.is_absolute() else project_root / output
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_video = self.output_dir / "staff_detection.mp4"
        self.output_csv = self.output_dir / "staff_detection.csv"
        print("Loading person detector...")
        self.person_model = YOLO(str(project_root / PERSON_MODEL))

        print("Loading badge detector...")
        self.badge_model = YOLO(str(project_root / BADGE_MODEL))

        self.tracker = PersonTracker()

    # ---------------------------------------------------------
    # Person detection
    # ---------------------------------------------------------

    def detect_people(self, frame):
        result = self.person_model.predict(
            frame,
            classes=[0],
            conf=PERSON_CONF,
            imgsz=PERSON_IMGSZ,
            verbose=False,
        )[0]

        detections = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist(),
            )

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": float(box.conf.item()),
            })

        return detections

    # ---------------------------------------------------------
    # Prepare upper-body crops for badge detector
    # ---------------------------------------------------------

    def prepare_badge_crops(self, frame, tracks):
        height, width = frame.shape[:2]

        crops = []
        crop_tracks = []
        offsets = []

        for track in tracks:
            x1, y1, x2, y2 = track.bbox

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(width, x2)
            y2 = min(height, y2)

            person_w = x2 - x1
            person_h = y2 - y1

            if person_w < 15 or person_h < 20:
                continue

            # Upper-body / torso region
            torso_x1 = x1 + int(person_w * 0.05)
            torso_x2 = x1 + int(person_w * 0.95)

            torso_y1 = y1
            torso_y2 = y1 + int(person_h * 0.75)

            crop = frame[
                torso_y1:torso_y2,
                torso_x1:torso_x2,
            ]

            if crop.size == 0:
                continue

            crops.append(crop)
            crop_tracks.append(track)

            offsets.append(
                (torso_x1, torso_y1)
            )

        return crops, crop_tracks, offsets

    # ---------------------------------------------------------
    # Badge detection
    #
    # IMPORTANT:
    # This function returns CURRENT-FRAME badge results only.
    # No staff state is saved inside the tracker.
    # ---------------------------------------------------------

    def detect_badges(self, frame, tracks):
        crops, crop_tracks, offsets = (
            self.prepare_badge_crops(
                frame,
                tracks,
            )
        )

        results_by_track = {}

        # Default every visible person to PERSON
        for track in tracks:
            results_by_track[track.id] = {
                "is_staff": False,
                "badge_conf": 0.0,
                "badge_box": None,
            }

        if not crops:
            return results_by_track

        results = self.badge_model.predict(
            crops,
            conf=BADGE_CONF,
            imgsz=BADGE_IMGSZ,
            verbose=False,
        )

        for track, offset, result in zip(
            crop_tracks,
            offsets,
            results,
        ):
            if result.boxes is None:
                continue

            best_box = None
            best_conf = 0.0

            for box in result.boxes:
                confidence = float(
                    box.conf.item()
                )

                if confidence > best_conf:
                    best_conf = confidence

                    best_box = tuple(
                        map(
                            int,
                            box.xyxy[0].tolist(),
                        )
                    )

            if best_box is None:
                continue

            offset_x, offset_y = offset

            bx1, by1, bx2, by2 = best_box

            global_box = (
                offset_x + bx1,
                offset_y + by1,
                offset_x + bx2,
                offset_y + by2,
            )

            results_by_track[track.id] = {
                "is_staff": True,
                "badge_conf": best_conf,
                "badge_box": global_box,
            }

        return results_by_track

    # ---------------------------------------------------------
    # Draw one person
    # ---------------------------------------------------------

    def draw_track(
        self,
        frame,
        track,
        badge_result,
    ):
        x1, y1, x2, y2 = map(
            int,
            track.bbox,
        )

        is_staff = badge_result[
            "is_staff"
        ]

        badge_conf = badge_result[
            "badge_conf"
        ]

        badge_box = badge_result[
            "badge_box"
        ]

        if is_staff:
            color = (0, 0, 255)

            label = (
                f"STAFF ID {track.id} "
                f"{badge_conf:.2f}"
            )

        elif track.is_static_suspect():
            color = (128, 128, 128)

            label = (
                f"STATIC? ID {track.id}"
            )

        else:
            color = (0, 255, 0)

            label = (
                f"PERSON ID {track.id}"
            )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        cv2.putText(
            frame,
            label,
            (
                x1,
                max(20, y1 - 8),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
        )

        # Draw badge only when detected in current frame
        if (
            is_staff
            and badge_box is not None
        ):
            bx1, by1, bx2, by2 = map(
                int,
                badge_box,
            )

            cv2.rectangle(
                frame,
                (bx1, by1),
                (bx2, by2),
                (255, 0, 255),
                2,
            )

            cv2.putText(
                frame,
                f"TAG {badge_conf:.2f}",
                (
                    bx1,
                    max(15, by1 - 4),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 0, 255),
                1,
            )

        # Bottom-centre coordinate
        x = int(
            (x1 + x2) / 2
        )

        y = y2

        if is_staff:
            cv2.circle(
                frame,
                (x, y),
                5,
                (255, 255, 0),
                -1,
            )

        return {
            "state": (
                "STAFF"
                if is_staff
                else "PERSON"
            ),
            "badge_conf": badge_conf,
            "x": x,
            "y": y,
        }

    # ---------------------------------------------------------
    # Main processing loop
    # ---------------------------------------------------------

    def run(self, max_frames=None):
        cap = cv2.VideoCapture(
            str(self.video_path)
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open {self.video_path}"
            )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )
        if fps <= 0:
            cap.release()
            raise RuntimeError("Input video has no usable frame rate")

        writer = cv2.VideoWriter(
            str(self.output_video),
            cv2.VideoWriter_fourcc(
                *"mp4v"
            ),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot write {self.output_video}")

        csv_file = open(
            self.output_csv,
            "w",
            newline="",
        )

        csv_writer = csv.writer(
            csv_file
        )

        csv_writer.writerow([
            "frame",
            "timestamp",
            "track_id",
            "state",
            "person_confidence",
            "badge_confidence",
            "x",
            "y",
        ])

        start_time = time.time()
        frame_idx = 0

        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, frame = cap.read()

            if not ok:
                break

            timestamp = (
                frame_idx / fps
            )

            # ---------------------------------------------
            # Step 1: Detect people
            # ---------------------------------------------

            detections = (
                self.detect_people(
                    frame
                )
            )

            # ---------------------------------------------
            # Step 2: Track people
            #
            # Tracking is ONLY for ID and bbox continuity.
            # It does not decide STAFF / PERSON.
            # ---------------------------------------------

            tracks = (
                self.tracker.update(
                    detections,
                    frame_idx,
                )
            )

            visible_tracks = [
                track
                for track in tracks
                if track.missed == 0
            ]

            # ---------------------------------------------
            # Step 3: Detect badges in CURRENT frame
            # ---------------------------------------------

            badge_results = (
                self.detect_badges(
                    frame,
                    visible_tracks,
                )
            )

            # ---------------------------------------------
            # Step 4: Draw and export
            # ---------------------------------------------

            annotated = frame.copy()

            for track in visible_tracks:
                badge_result = (
                    badge_results.get(
                        track.id,
                        {
                            "is_staff": False,
                            "badge_conf": 0.0,
                            "badge_box": None,
                        },
                    )
                )

                result = (
                    self.draw_track(
                        annotated,
                        track,
                        badge_result,
                    )
                )

                csv_writer.writerow([
                    frame_idx,
                    round(
                        timestamp,
                        3,
                    ),
                    track.id,
                    result["state"],
                    round(
                        track.conf,
                        4,
                    ),
                    round(
                        result[
                            "badge_conf"
                        ],
                        4,
                    ),
                    result["x"],
                    result["y"],
                ])

            cv2.putText(
                annotated,
                f"Frame {frame_idx}",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            writer.write(
                annotated
            )

            # ---------------------------------------------
            # Progress
            # ---------------------------------------------

            if frame_idx % 100 == 0:
                elapsed = (
                    time.time()
                    - start_time
                )

                speed = (
                    (frame_idx + 1)
                    / elapsed
                )

                print(
                    f"Frame "
                    f"{frame_idx}/"
                    f"{total_frames} "
                    f"| "
                    f"{speed:.1f} FPS"
                )

            frame_idx += 1

        cap.release()
        writer.release()
        csv_file.close()
        (self.output_dir / "run_metadata.json").write_text(
            json.dumps({"video": str(self.video_path), "frame_count": frame_idx, "fps": fps}, indent=2),
            encoding="utf-8",
        )

        elapsed = (
            time.time()
            - start_time
        )

        duration = (
            frame_idx / fps
        )

        effective_fps = (
            frame_idx / elapsed
        )

        print()
        print("Finished")

        print(
            f"Video: {self.output_video}"
        )

        print(
            f"CSV: {self.output_csv}"
        )

        print(
            f"Effective FPS: "
            f"{effective_fps:.2f}"
        )

        print(
            "Realtime factor: "
            + (f"{elapsed / duration:.2f}x" if duration else "n/a (zero frames)")
        )
