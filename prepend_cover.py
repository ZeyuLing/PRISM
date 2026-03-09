#!/usr/bin/env python3
"""
Prepend a 3-second cover to an existing video, preserving the original audio.

Use this when the input video already has an audio track (e.g. voiceover) that
should not be lost. Unlike compose_demo_video.py, this script does NOT re-encode
the main content — it only prepends a cover and concatenates, keeping the original
audio (with 3s silence for the cover).

Usage:
  python prepend_cover.py -i prism_presentation_demo.mp4 -o prism_promo.mp4
"""

import os
import sys
import subprocess
import tempfile
import argparse

# Add compose_demo_video's directory for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import cv2

# Import cover rendering from compose_demo_video
from compose_demo_video import (
    FIG_TEASER,
    load_figure,
    load_cover_fonts,
    render_cover_frame,
)


def get_ffmpeg():
    """Get ffmpeg executable path (imageio_ffmpeg or system)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    return "ffmpeg"


def probe_audio_format(video_path):
    """Get sample rate and channel layout from video. Returns (sample_rate, channel_layout)."""
    for ffprobe in ["ffprobe", "avprobe"]:
        try:
            cmd = [
                ffprobe, "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate,channel_layout",
                "-of", "csv=p=0",
                video_path,
            ]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if out.returncode == 0 and out.stdout.strip():
                parts = out.stdout.strip().split(",")
                rate = int(float(parts[0])) if parts and parts[0] else 44100
                layout = parts[1] if len(parts) > 1 and parts[1] else "stereo"
                return rate, layout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return 44100, "stereo"  # fallback


def main():
    parser = argparse.ArgumentParser(
        description="Prepend 3s cover to video, preserving audio"
    )
    parser.add_argument("-i", "--input", required=True, help="Input video (with audio)")
    parser.add_argument("-o", "--output", required=True, help="Output video path")
    parser.add_argument("-d", "--cover-duration", type=float, default=3.0,
                        help="Cover duration in seconds (default: 3)")
    args = parser.parse_args()

    ffmpeg_bin = get_ffmpeg()
    if not ffmpeg_bin:
        print("ERROR: ffmpeg not found. Install imageio-ffmpeg or ffmpeg.")
        sys.exit(1)

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not os.path.exists(input_path):
        print(f"ERROR: Input not found: {input_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(input_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print(f"Input: {input_path} ({W}x{H} @ {fps:.1f}fps)")
    print(f"Output: {output_path}")

    # Load figures and fonts for cover (large fonts)
    figures = {}
    if os.path.exists(FIG_TEASER):
        figures["teaser_cover"] = load_figure(FIG_TEASER, int(W * 0.92), int(H * 0.62))
    fonts = load_cover_fonts(W, H)

    # Render cover frame
    print("Rendering cover frame...")
    cover_img = render_cover_frame(W, H, fonts, figures)
    cover_path = tempfile.mktemp(suffix=".png")
    cover_img.save(cover_path)

    cover_dur = args.cover_duration
    cover_video = tempfile.mktemp(suffix="_cover.mp4")
    concat_list = None
    try:
        # Probe original audio format for sync
        sample_rate, channel_layout = probe_audio_format(input_path)

        # Cover video: image looped for cover_dur, NO audio
        cmd_cover = [
            ffmpeg_bin, "-y",
            "-loop", "1", "-i", cover_path,
            "-t", str(cover_dur),
            "-vf", f"scale={W}:{H}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-r", str(fps),
            "-an",
            cover_video,
        ]
        print("Creating cover video (no audio)...")
        subprocess.run(cmd_cover, check=True, capture_output=True)

        # Concat: [cover_video] + [original] using filter_complex
        # Video: cover + original
        # Audio: adelay prepends silence so original audio starts when original video starts
        delay_ms = int(cover_dur * 1000)
        delay_str = f"{delay_ms}|{delay_ms}"  # stereo
        filter_complex = (
            f"[0:v][1:v]concat=n=2:v=1:a=0[outv];"
            f"[1:a]adelay={delay_ms}|{delay_ms}[outa]"
        )

        cmd_concat = [
            ffmpeg_bin, "-y",
            "-i", cover_video,
            "-i", input_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]
        print("Concatenating cover + original (audio aligned)...")
        subprocess.run(cmd_concat, check=True, capture_output=True)

        print(f"Done! Output: {output_path}")
    finally:
        to_remove = [cover_path, cover_video]
        if concat_list and os.path.exists(concat_list):
            to_remove.append(concat_list)
        for p in to_remove:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


if __name__ == "__main__":
    main()
