import argparse

from config import OUTPUT_DIR, VIDEO_PATH
from pipeline import StaffDetectionPipeline


def main():
    parser = argparse.ArgumentParser(description="Detect staff in overhead CCTV video.")
    parser.add_argument("--video", default=VIDEO_PATH, help="Input video (default: sample.mp4)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for video and CSV outputs")
    parser.add_argument("--max-frames", type=int, help="Process only this many frames (for a smoke test)")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    pipeline = StaffDetectionPipeline(video_path=args.video, output_dir=args.output_dir)
    pipeline.run(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
