import math
from collections import deque

import numpy as np

from config import (
    MAX_MISSED_FRAMES,
    STATIC_CHECK_MIN_AGE,
    STATIC_MAX_AVG_CONF,
    STATIC_MAX_SPREAD,
    TRACK_MAX_CENTER_DISTANCE,
    TRACK_MIN_IOU,
)


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def bbox_center(box):
    x1, y1, x2, y2 = box

    return np.array([
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    ])


def bbox_size(box):
    x1, y1, x2, y2 = box

    return (
        max(1, x2 - x1),
        max(1, y2 - y1),
    )


class Track:
    def __init__(self, track_id, detection, frame_idx):
        self.id = track_id

        self.bbox = detection["bbox"]
        self.conf = detection["conf"]

        self.age = 1
        self.missed = 0
        self.last_frame = frame_idx

        self.velocity = np.zeros(2, dtype=float)

        self.centers = deque(maxlen=100)
        self.centers.append(
            bbox_center(self.bbox)
        )

        self.conf_history = deque(maxlen=100)
        self.conf_history.append(
            self.conf
        )

    def predicted_bbox(self):
        x1, y1, x2, y2 = self.bbox
        dx, dy = self.velocity

        return (
            int(x1 + dx),
            int(y1 + dy),
            int(x2 + dx),
            int(y2 + dy),
        )

    def update(self, detection, frame_idx):
        old_center = bbox_center(
            self.bbox
        )

        new_bbox = detection["bbox"]
        new_center = bbox_center(
            new_bbox
        )

        movement = (
            new_center - old_center
        )

        self.velocity = (
            0.7 * self.velocity
            + 0.3 * movement
        )

        self.bbox = new_bbox
        self.conf = detection["conf"]

        self.centers.append(
            new_center
        )

        self.conf_history.append(
            self.conf
        )

        self.age += 1
        self.missed = 0
        self.last_frame = frame_idx

    def mark_missed(self):
        self.missed += 1

    def is_static_suspect(self):
        if self.age < STATIC_CHECK_MIN_AGE:
            return False

        if len(self.centers) < 20:
            return False

        points = np.array(
            self.centers
        )

        x_range = (
            points[:, 0].max()
            - points[:, 0].min()
        )

        y_range = (
            points[:, 1].max()
            - points[:, 1].min()
        )

        movement_spread = math.hypot(
            x_range,
            y_range,
        )

        avg_conf = float(
            np.mean(
                self.conf_history
            )
        )

        return (
            movement_spread < STATIC_MAX_SPREAD
            and
            avg_conf < STATIC_MAX_AVG_CONF
        )

class PersonTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, frame_idx):
        unmatched_detections = set(
            range(len(detections))
        )

        unmatched_tracks = set(
            self.tracks.keys()
        )

        candidates = []

        for track_id, track in self.tracks.items():
            predicted = (
                track.predicted_bbox()
            )

            predicted_center = (
                bbox_center(predicted)
            )

            width, height = (
                bbox_size(predicted)
            )

            diagonal = max(
                math.hypot(
                    width,
                    height,
                ),
                1.0,
            )

            for det_idx, detection in enumerate(
                detections
            ):
                box = detection["bbox"]

                iou = bbox_iou(
                    predicted,
                    box,
                )

                distance = np.linalg.norm(
                    predicted_center
                    - bbox_center(box)
                )

                normalized_distance = (
                    distance / diagonal
                )

                if (
                    iou >= TRACK_MIN_IOU
                    or
                    normalized_distance
                    <= TRACK_MAX_CENTER_DISTANCE
                ):
                    cost = (
                        normalized_distance
                        - 0.8 * iou
                    )

                    candidates.append(
                        (
                            cost,
                            track_id,
                            det_idx,
                        )
                    )

        candidates.sort(
            key=lambda item: item[0]
        )

        for _, track_id, det_idx in candidates:
            if (
                track_id
                not in unmatched_tracks
            ):
                continue

            if (
                det_idx
                not in unmatched_detections
            ):
                continue

            self.tracks[
                track_id
            ].update(
                detections[det_idx],
                frame_idx,
            )

            unmatched_tracks.remove(
                track_id
            )

            unmatched_detections.remove(
                det_idx
            )

        # Missed tracks
        for track_id in list(
            unmatched_tracks
        ):
            track = self.tracks[
                track_id
            ]

            track.mark_missed()

            if (
                track.missed
                > MAX_MISSED_FRAMES
            ):
                del self.tracks[
                    track_id
                ]

        # New tracks
        for det_idx in unmatched_detections:
            track = Track(
                self.next_id,
                detections[det_idx],
                frame_idx,
            )

            self.tracks[
                self.next_id
            ] = track

            self.next_id += 1

        return list(
            self.tracks.values()
        )