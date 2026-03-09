#!/usr/bin/env python3
"""
Compose a presentation video for PRISM paper demo.

Overlays on top of the demo motion video:
1. Title card with paper name and key insight
2. Current action prompt as subtitle (bottom bar)
3. Segment counter
4. Paper figures as picture-in-picture at strategic moments
5. Contribution text overlays
6. Key results overlay near the end
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PAPER_DIR = os.path.join(_REPO_ROOT, "papers", "PRISM_ECCV2026")
FIG_TEASER = os.path.join(PAPER_DIR, "figures", "fig_teaser.png")
FIG_PIPELINE = os.path.join(PAPER_DIR, "figures", "fig_pipeline.png")

FONT_REGULAR = "/usr/share/fonts/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"

# ── Default demo segments (overridden by --info) ──────────────────
SEGMENTS = None  # loaded from info.json at runtime


def load_segments_from_info(info_path):
    """Load segment prompts and frame counts from an info.json file."""
    import json
    with open(info_path) as f:
        info = json.load(f)
    prompts = info["prompts"]
    num_frames = info["num_frames_per_segment"]
    return [{"prompt": p, "num_frames": n} for p, n in zip(prompts, num_frames)]

# ── Narration script: (start_s, end_s, display_text, voiceover_text) ──
# display_text: short keywords shown on screen
# voiceover_text: complete sentences for voice narration
NARRATION_SCRIPT = [
    (0.0, 6.0,
     "Unified Foundation Model for 3D Human Motion",
     "Welcome! We present PRISM \u2014 a unified model "
     "for text-driven, pose-conditioned, and streaming motion generation."),
    (7.0, 12.0,
     "Joint-Factorized Motion Latent Space",
     "Its key idea is a joint-factorized latent space "
     "that gives each body joint its own dedicated token."),
    (12.5, 20.0,
     "Noise-Free Condition Injection",
     "This representation enables noise-free condition injection \u2014 "
     "letting one model handle all three tasks "
     "with no architectural changes."),
    (20.5, 29.0,
     "Self-Forcing for Long-Horizon Stability",
     "For long sequences, self-forcing training prevents drift "
     "by feeding the model\u2019s own outputs back during training."),
    (30.0, 39.0,
     "21 Segments \u2014 Seamless Streaming",
     "As you can see, twenty-one diverse segments are generated "
     "seamlessly \u2014 each one closely follows its text prompt \u2014 "
     "with smooth transitions and no quality loss."),
    (40.0, 49.0,
     "State-of-the-Art Results",
     "PRISM achieves state-of-the-art for text-to-motion "
     "on HumanML3D and MotionHub, and for sequential composition on BABEL. "
     "We run a user study on narrative composition, "
     "and users prefer PRISM in over seventy percent of comparisons."),
    (50.0, 59.0,
     "1.4B Parameters \u00b7 Unlimited Length",
     "The full model has 1.4 billion parameters, "
     "trained on 200K pairs, generating motion at unlimited length."),
    (60.0, 74.3,
     "One Model.  Infinite Motion.",
     "That\u2019s PRISM in a nutshell \u2014 one model, infinite motion. "
     "Thank you for watching."),
]

# ── Colors ─────────────────────────────────────────────────────────
TEAL = (0, 137, 123)
DARK_BG = (30, 30, 35)
WHITE = (255, 255, 255)
LIGHT_GRAY = (200, 200, 200)
ACCENT_ORANGE = (255, 138, 101)
ACCENT_GOLD = (255, 193, 7)

# Cover page: white background, dark text
COVER_BG = (255, 255, 255)
COVER_TITLE_COLOR = (25, 25, 30)
COVER_SUBTITLE_COLOR = (80, 80, 90)
COVER_AUTHORS_COLOR = (40, 40, 50)
COVER_SMALL_COLOR = (90, 90, 100)


def load_fonts(scale=1.0):
    """Load fonts at appropriate sizes for the video resolution."""
    s = scale
    return {
        "title": ImageFont.truetype(FONT_BOLD, int(72 * s)),
        "subtitle": ImageFont.truetype(FONT_REGULAR, int(48 * s)),
        "heading": ImageFont.truetype(FONT_BOLD, int(44 * s)),
        "body": ImageFont.truetype(FONT_REGULAR, int(36 * s)),
        "small": ImageFont.truetype(FONT_REGULAR, int(28 * s)),
        "segment_num": ImageFont.truetype(FONT_BOLD, int(36 * s)),
        "big_title": ImageFont.truetype(FONT_BOLD, int(96 * s)),
        "medium_bold": ImageFont.truetype(FONT_BOLD, int(40 * s)),
    }


def load_cover_fonts(W, H):
    """Load fonts at LARGE sizes for cover (readable on full-screen)."""
    scale = min(W, H) / 960
    scale = max(scale, 1.8)
    s = scale
    return {
        "big_title": ImageFont.truetype(FONT_BOLD, int(140 * s)),
        "subtitle": ImageFont.truetype(FONT_REGULAR, int(42 * s)),
        "body": ImageFont.truetype(FONT_REGULAR, int(36 * s)),
        "small": ImageFont.truetype(FONT_REGULAR, int(30 * s)),
    }


def load_figure(path, max_width, max_height):
    """Load a figure image and resize to fit within bounds.

    Transparent PNGs are composited onto a white background so they
    don't look washed-out when overlaid on the video.
    """
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    ratio = min(max_width / w, max_height / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    bg = Image.new("RGBA", (new_w, new_h), (255, 255, 255, 255))
    img = Image.alpha_composite(bg, img)
    return img


def draw_rounded_rect(draw, xy, fill, radius=12):
    """Draw a filled rounded rectangle."""
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill)


def alpha_composite_at(base, overlay, position):
    """Paste an RGBA overlay onto an RGBA base at the given position."""
    x, y = position
    temp = Image.new("RGBA", base.size, (0, 0, 0, 0))
    temp.paste(overlay, (x, y))
    return Image.alpha_composite(base, temp)


def ease_in_out(t):
    """Smooth ease-in-out curve, t in [0, 1]."""
    return t * t * (3 - 2 * t)


def compute_alpha(frame_time, start, end, fade_dur=0.6):
    """Compute overlay alpha with fade in/out."""
    if frame_time < start or frame_time > end:
        return 0.0
    if frame_time < start + fade_dur:
        return ease_in_out((frame_time - start) / fade_dur)
    if frame_time > end - fade_dur:
        return ease_in_out((end - frame_time) / fade_dur)
    return 1.0


def build_segment_boundaries(segments, video_total_frames):
    """Compute (start_frame, end_frame) in video space for each segment.

    Segments share 1-frame overlap at each boundary.  The resulting
    actual total is sum(num_frames) - (N-1).  We then linearly scale
    these boundaries to the video's total frame count.
    """
    n = len(segments)
    cum = [0]
    for i, seg in enumerate(segments):
        overlap = 1 if i > 0 else 0
        cum.append(cum[-1] + seg["num_frames"] - overlap)
    actual_total = cum[-1]
    scale = video_total_frames / actual_total
    boundaries = []
    for i in range(n):
        s = int(round(cum[i] * scale))
        e = int(round(cum[i + 1] * scale))
        boundaries.append((s, e))
    return boundaries


def get_segment_index_from_boundaries(frame_idx, boundaries):
    """Binary-search style lookup into precomputed boundaries."""
    for i, (s, e) in enumerate(boundaries):
        if frame_idx < e:
            return i
    return len(boundaries) - 1


def render_text_block(draw, text, xy, font, fill=(255, 255, 255, 255), max_width=None):
    """Render text, optionally wrapping to max_width. Returns bounding box height."""
    if max_width is None:
        draw.text(xy, text, font=font, fill=fill)
        bbox = draw.textbbox(xy, text, font=font)
        return bbox[3] - bbox[1]

    words = text.split()
    lines = []
    current = ""
    for w in words:
        test = f"{current} {w}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > max_width and current:
            lines.append(current)
            current = w
        else:
            current = test
    if current:
        lines.append(current)

    x, y = xy
    total_h = 0
    for line in lines:
        draw.text((x, y + total_h), line, font=font, fill=fill)
        bbox = draw.textbbox((0, 0), line, font=font)
        total_h += int((bbox[3] - bbox[1]) * 1.3)
    return total_h


# ── Cover page for promo video (title, authors, teaser, 3s) ─────────
# First three authors are co-first (共一)
COVER_AUTHORS = "Zeyu Ling*, Qing Shuai*, Teng Zhang*, Shiyang Li, Bo Han, Changqing Zou"
COVER_EQUAL_CONTRIB = "*Equal contribution"
COVER_AFFILIATIONS = (
    "State Key Lab of CAD & CG, Zhejiang University  \u2022  "
    "Zhejiang Lab  \u2022  Computer Animation & Perception Group, ZJU  \u2022  "
    "VIVID Team, ByteDance"
)


def render_cover_frame(W, H, fonts, figures):
    """Render a single cover frame: title, authors, teaser figure.
    Layout: title at top, gap, teaser, authors at bottom.
    White background, dark text. Uses teaser_cover if available.
    """
    pad = int(min(W, H) * 0.05)
    title_fig_gap = int(H * 0.07)  # Gap between title block and teaser
    bg = Image.new("RGB", (W, H), COVER_BG)
    base = bg.convert("RGBA")

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Title block at top (centered)
    title_y = pad
    bbox = draw.textbbox((0, 0), "PRISM", font=fonts["big_title"])
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, title_y), "PRISM", font=fonts["big_title"], fill=COVER_TITLE_COLOR)
    title_y += int((bbox[3] - bbox[1]) * 1.6)  # Larger gap between PRISM and subtitle
    sub = "Streaming Human Motion Generation with Per-Joint Latent Decomposition"
    bbox = draw.textbbox((0, 0), sub, font=fonts["subtitle"])
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, title_y), sub, font=fonts["subtitle"], fill=COVER_SUBTITLE_COLOR)
    title_block_bottom = title_y + int((bbox[3] - bbox[1]) * 1.3)

    # Teaser: positioned below title with gap, above authors
    auth_block_top = H - pad - 140
    fig_area_h = auth_block_top - title_block_bottom - title_fig_gap
    fig_area_w = int(W * 0.90)
    fig = figures.get("teaser_cover") or figures.get("teaser")
    if fig is not None and fig_area_h > 0:
        fw, fh = fig.size
        scale = min(fig_area_w / fw, fig_area_h / fh)
        fw, fh = int(fw * scale), int(fh * scale)
        fig = fig.resize((fw, fh), Image.LANCZOS)
        fig_x = (W - fw) // 2
        fig_y = title_block_bottom + title_fig_gap
        base = alpha_composite_at(base, fig.convert("RGBA"), (fig_x, fig_y))

    # Authors at bottom (centered), with equal contribution note
    auth_y = H - pad - 120
    bbox = draw.textbbox((0, 0), COVER_AUTHORS, font=fonts["body"])
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, auth_y), COVER_AUTHORS, font=fonts["body"], fill=COVER_AUTHORS_COLOR)
    auth_y += int((bbox[3] - bbox[1]) * 1.1)
    bbox = draw.textbbox((0, 0), COVER_EQUAL_CONTRIB, font=fonts["small"])
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, auth_y), COVER_EQUAL_CONTRIB, font=fonts["small"], fill=COVER_SMALL_COLOR)
    auth_y += int((bbox[3] - bbox[1]) * 1.2)
    bbox = draw.textbbox((0, 0), COVER_AFFILIATIONS, font=fonts["small"])
    aff_w = bbox[2] - bbox[0]
    max_w = int(W * 0.9)
    if aff_w > max_w:
        words = COVER_AFFILIATIONS.split()
        lines, cur = [], ""
        for w in words:
            test = f"{cur} {w}".strip() if cur else w
            b = draw.textbbox((0, 0), test, font=fonts["small"])
            if b[2] - b[0] > max_w and cur:
                lines.append(cur)
                cur = w
            else:
                cur = test
        if cur:
            lines.append(cur)
        for line in lines:
            b = draw.textbbox((0, 0), line, font=fonts["small"])
            draw.text(((W - (b[2] - b[0])) // 2, auth_y), line, font=fonts["small"], fill=COVER_SMALL_COLOR)
            auth_y += int((b[3] - b[1]) * 1.2)
    else:
        draw.text(((W - aff_w) // 2, auth_y), COVER_AFFILIATIONS, font=fonts["small"], fill=COVER_SMALL_COLOR)

    result = Image.alpha_composite(base, overlay)
    return result.convert("RGB")


def compose_frame(pil_frame, frame_idx, total_frames, fps, fonts, figures,
                  segments, boundaries, subtitle_only=False):
    """Add all overlays to a single video frame."""
    W, H = pil_frame.size
    t = frame_idx / fps
    duration = total_frames / fps

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    seg_idx = get_segment_index_from_boundaries(frame_idx, boundaries)
    prompt = segments[seg_idx]["prompt"]
    num_segments = len(segments)

    # ── Bottom segment bar (shared by all modes) ────────────────────
    def _draw_segment_bar():
        sub_alpha = compute_alpha(t, 0.5, duration - 0.5, fade_dur=0.6)
        if sub_alpha <= 0:
            return
        sa = int(170 * sub_alpha)
        bar_y = H - 130
        draw_rounded_rect(draw, (60, bar_y, W - 60, H - 30),
                          fill=(*DARK_BG, sa), radius=16)
        ta = int(255 * sub_alpha)
        seg_text = f"Segment {seg_idx + 1} / {num_segments}"
        draw.text((100, bar_y + 18), seg_text,
                  font=fonts["segment_num"], fill=(*TEAL, ta))
        seg_bbox = draw.textbbox((0, 0), seg_text, font=fonts["segment_num"])
        seg_w = seg_bbox[2] - seg_bbox[0]
        draw.text((100 + seg_w + 40, bar_y + 22),
                  f'\u201c{prompt}\u201d', font=fonts["subtitle"], fill=(*WHITE, ta))
        prog_y = bar_y + 80
        prog_w = W - 220
        draw_rounded_rect(draw, (110, prog_y, 110 + prog_w, prog_y + 8),
                          fill=(80, 80, 80, int(120 * sub_alpha)), radius=4)
        filled = int(prog_w * (t / duration))
        if filled > 0:
            draw_rounded_rect(draw, (110, prog_y, 110 + filled, prog_y + 8),
                              fill=(*TEAL, int(220 * sub_alpha)), radius=4)

    if subtitle_only:
        _draw_segment_bar()
        base_rgba = pil_frame.convert("RGBA")
        return Image.alpha_composite(base_rgba, overlay).convert("RGB")

    # ══════════════════════════════════════════════════════════════
    # NARRATION MODE: title + top narration + figures + bottom bar
    # ══════════════════════════════════════════════════════════════

    NARRATIONS = NARRATION_SCRIPT

    # ── 1. Title card (0-6s): prominent paper title ───────────────
    title_alpha = compute_alpha(t, 0, 6.0, fade_dur=0.8)
    if title_alpha > 0:
        a = int(190 * title_alpha)
        bar_h = 220
        draw_rounded_rect(draw, (60, 30, W - 60, 30 + bar_h),
                          fill=(*DARK_BG, a), radius=20)
        ta = int(255 * title_alpha)
        draw.text((120, 50), "PRISM", font=fonts["big_title"],
                  fill=(*WHITE, ta))
        bbox = draw.textbbox((0, 0), "PRISM", font=fonts["big_title"])
        prism_w = bbox[2] - bbox[0]
        draw.text((120 + prism_w + 30, 80),
                  "Per-joint Representation for Infinite Streaming Motion",
                  font=fonts["subtitle"], fill=(*LIGHT_GRAY, ta))
        draw.text((120, 165),
                  "Text-to-Motion  \u2022  Pose-Conditioned  \u2022  "
                  "Streaming Generation  \u2022  Narrative Composition",
                  font=fonts["body"], fill=(*ACCENT_GOLD, ta))

    # ── 2. Display text banner (short keywords, top area) ──────
    for n_start, n_end, n_display, _n_voice in NARRATIONS:
        if n_display is None:
            continue
        na = compute_alpha(t, n_start, n_end, fade_dur=0.7)
        if na <= 0:
            continue
        bar_top = 270 if title_alpha > 0 else 35
        bbox = draw.textbbox((0, 0), n_display, font=fonts["heading"])
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        pad_x, pad_y = 50, 20
        block_w = text_w + 2 * pad_x
        block_h = text_h + 2 * pad_y
        bx = (W - block_w) // 2
        a = int(185 * na)
        draw_rounded_rect(draw, (bx, bar_top, bx + block_w, bar_top + block_h),
                          fill=(*DARK_BG, a), radius=16)
        ta = int(255 * na)
        draw.text(((W - text_w) // 2, bar_top + pad_y), n_display,
                  font=fonts["heading"], fill=(*ACCENT_GOLD, ta))
        break  # only one display text active at a time

    # ── 3. Teaser figure (7-12s, right side) ─────────────────────
    teaser_alpha = compute_alpha(t, 7.0, 12.0, fade_dur=0.8)
    if teaser_alpha > 0 and figures.get("teaser") is not None:
        fig = figures["teaser"]
        fw, fh = fig.size
        fig_x = W - fw - 80
        fig_y = 280
        a = int(200 * teaser_alpha)
        draw_rounded_rect(draw,
                          (fig_x - 15, fig_y - 10, fig_x + fw + 15, fig_y + fh + 45),
                          fill=(*DARK_BG, a), radius=14)
        draw.text((fig_x, fig_y + fh + 8),
                  "Fig 1: Tasks supported by PRISM",
                  font=fonts["small"],
                  fill=(*LIGHT_GRAY, int(230 * teaser_alpha)))
        fig_a = fig.copy()
        if fig_a.mode == "RGBA":
            r, g, b, oa = fig_a.split()
            fig_a = Image.merge("RGBA", (r, g, b,
                                         oa.point(lambda x: int(x * teaser_alpha))))
        else:
            fig_a.putalpha(int(255 * teaser_alpha))
        overlay = alpha_composite_at(overlay, fig_a, (fig_x, fig_y))
        draw = ImageDraw.Draw(overlay)

    # ── 4. Pipeline figure (12.5-29s, right side) ────────────────
    pipe_alpha = compute_alpha(t, 12.5, 29.0, fade_dur=1.0)
    if pipe_alpha > 0 and figures.get("pipeline") is not None:
        fig = figures["pipeline"]
        fw, fh = fig.size
        fig_x = W - fw - 80
        fig_y = 280
        a = int(210 * pipe_alpha)
        draw_rounded_rect(draw,
                          (fig_x - 15, fig_y - 10, fig_x + fw + 15, fig_y + fh + 45),
                          fill=(*DARK_BG, a), radius=14)
        draw.text((fig_x, fig_y + fh + 8),
                  "Fig 2: Joint-Factorized VAE + Noise-Free Condition Injection",
                  font=fonts["small"],
                  fill=(*LIGHT_GRAY, int(230 * pipe_alpha)))
        fig_a = fig.copy()
        if fig_a.mode == "RGBA":
            r, g, b, oa = fig_a.split()
            fig_a = Image.merge("RGBA", (r, g, b,
                                         oa.point(lambda x: int(x * pipe_alpha))))
        else:
            fig_a.putalpha(int(255 * pipe_alpha))
        overlay = alpha_composite_at(overlay, fig_a, (fig_x, fig_y))
        draw = ImageDraw.Draw(overlay)

    # ── 5. Bottom segment bar ────────────────────────────────────
    _draw_segment_bar()

    # Composite overlay onto base frame
    base_rgba = pil_frame.convert("RGBA")
    result = Image.alpha_composite(base_rgba, overlay)
    return result.convert("RGB")


def main(input_video, output_video=None, subtitle_only=False, info_json=None, with_cover=False):
    global SEGMENTS

    if info_json:
        SEGMENTS = load_segments_from_info(info_json)
        print(f"Loaded {len(SEGMENTS)} segments from {info_json}")
    elif SEGMENTS is None:
        print("ERROR: No --info provided and no built-in segments. Use --info <info.json>.")
        sys.exit(1)

    segments = SEGMENTS

    if output_video is None:
        base, ext = os.path.splitext(input_video)
        suffix = "_subtitle" if subtitle_only else "_promo" if with_cover else "_presentation"
        output_video = base + suffix + ".mp4"

    print(f"Input:  {input_video}")
    print(f"Output: {output_video}")

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video:  {W}x{H} @ {fps:.1f}fps, {total_frames} frames ({total_frames/fps:.1f}s)")

    boundaries = build_segment_boundaries(segments, total_frames)
    print("Segment boundaries (video frames):")
    for i, (s, e) in enumerate(boundaries):
        t_s, t_e = s / fps, e / fps
        print(f"  Seg {i+1:2d}: frame {s:5d}–{e:5d}  "
              f"({t_s:5.1f}s–{t_e:5.1f}s, {t_e-t_s:.1f}s)  "
              f"{segments[i]['prompt']}")

    fonts = load_fonts(scale=W / 3456)

    figures = {}
    if not subtitle_only:
        print("Loading figures...")
        max_fig_w = int(W * 0.45)
        max_fig_h = int(H * 0.45)
        if os.path.exists(FIG_TEASER):
            figures["teaser"] = load_figure(FIG_TEASER, max_fig_w, max_fig_h)
            print(f"  teaser: {figures['teaser'].size}")
            if with_cover:
                figures["teaser_cover"] = load_figure(FIG_TEASER, int(W * 0.92), int(H * 0.62))
                print(f"  teaser_cover: {figures['teaser_cover'].size}")
        if os.path.exists(FIG_PIPELINE):
            figures["pipeline"] = load_figure(FIG_PIPELINE, max_fig_w, max_fig_h)
            print(f"  pipeline: {figures['pipeline'].size}")

    if not subtitle_only:
        print("\n── Voiceover Narration Timeline ──")
        for ns, ne, _disp, voice in NARRATION_SCRIPT:
            m_s, s_s = divmod(int(ns), 60)
            m_e, s_e = divmod(int(ne), 60)
            print(f"  [{m_s:d}:{s_s:02d} – {m_e:d}:{s_e:02d}]  {voice}")
        print()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (W, H))

    cover_duration = 3.0
    cover_frames = int(cover_duration * fps) if with_cover else 0

    if with_cover and not subtitle_only:
        print(f"Writing cover ({cover_duration}s, {cover_frames} frames)...")
        cover_fonts = load_cover_fonts(W, H)
        cover_img = render_cover_frame(W, H, cover_fonts, figures)
        cover_bgr = cv2.cvtColor(np.array(cover_img), cv2.COLOR_RGB2BGR)
        for _ in range(cover_frames):
            out.write(cover_bgr)

    print("Processing frames...")
    frame_idx = 0
    report_interval = max(1, total_frames // 20)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        composed = compose_frame(pil_frame, frame_idx, total_frames, fps, fonts, figures,
                                segments, boundaries, subtitle_only=subtitle_only)
        out_frame = cv2.cvtColor(np.array(composed), cv2.COLOR_RGB2BGR)
        out.write(out_frame)

        if frame_idx % report_interval == 0:
            pct = frame_idx / total_frames * 100
            print(f"  [{pct:5.1f}%] frame {frame_idx}/{total_frames}")

        frame_idx += 1

    cap.release()
    out.release()
    print(f"\nDone! {frame_idx} frames written to {output_video}")

    # Re-encode with ffmpeg for better compatibility (H264)
    ffmpeg_bin = None
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    if ffmpeg_bin:
        final_output = output_video.replace(".mp4", "_h264.mp4")
        cmd = (f'{ffmpeg_bin} -y -i "{output_video}" '
               f'-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p '
               f'"{final_output}" 2>/dev/null')
        print(f"Re-encoding to H264: {final_output}")
        ret = os.system(cmd)
        if ret == 0:
            os.remove(output_video)
            os.rename(final_output, output_video)
            print(f"Final output: {output_video}")
        else:
            print(f"H264 re-encode failed (code {ret}), keeping mp4v version")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input demo video path")
    parser.add_argument("--output", "-o", default=None, help="Output video path")
    parser.add_argument("--info", default=None,
                        help="Path to info.json with segment prompts & frame counts")
    parser.add_argument("--subtitle-only", action="store_true",
                        help="Only render the bottom subtitle bar")
    parser.add_argument("--with-cover", action="store_true",
                        help="Prepend 3s cover (title, authors, teaser) for promo video")
    args = parser.parse_args()
    main(args.input, args.output, subtitle_only=args.subtitle_only,
         info_json=args.info, with_cover=args.with_cover)
