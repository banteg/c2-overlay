from __future__ import annotations

import argparse
import shutil
import sys
from datetime import timedelta
from pathlib import Path

from .align import choose_anchor_index
from .ffmpeg import burn_in
from .fit import parse_data_file
from .overlay import generate_ass
from .timeutil import parse_iso8601
from .video import get_video_metadata


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="c2-overlay",
        description="Create a PM5-style overlay (.ass subtitles) from a Concept2 FIT file and align it to a video using metadata timestamps.",
    )
    ap.add_argument("video", help="Input video file (mp4/mov/etc)")
    ap.add_argument("fit", help="Concept2 workout data file (.fit)")
    ap.add_argument(
        "-o",
        "--out-ass",
        default=None,
        help="Output .ass path (default: next to input video)",
    )

    ap.add_argument(
        "--video-start",
        default=None,
        help="Override the video's start timestamp (ISO-8601, e.g. 2025-12-14T10:41:31Z).",
    )
    ap.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Manual offset adjustment in seconds (added to the auto-computed alignment). "
        "Positive makes data appear later; negative earlier.",
    )
    ap.add_argument(
        "--font",
        default=None,
        help="Legacy alias: set both --label-font and --value-font to this font name.",
    )
    ap.add_argument(
        "--label-font",
        default="PragmataPro",
        help="Font for labels (must exist on your system)",
    )
    ap.add_argument(
        "--value-font",
        default="PragmataPro Mono",
        help="Font for values (must exist on your system)",
    )
    ap.add_argument(
        "--fontsize",
        type=int,
        default=None,
        help="Value font size (default: scaled from 52 @ 1080p)",
    )
    ap.add_argument(
        "--left-margin",
        type=int,
        default=None,
        help="Left margin in pixels (default: scaled from 20 @ 1080p)",
    )
    ap.add_argument(
        "--top-margin",
        type=int,
        default=None,
        help="Top margin in pixels; if set, positions the overlay from the top instead of the bottom.",
    )
    ap.add_argument(
        "--bottom-margin",
        type=int,
        default=None,
        help="Bottom margin in pixels (default: scaled from 20 @ 1080p)",
    )
    ap.add_argument(
        "--box-alpha",
        type=int,
        default=112,
        help="Background box transparency 0..255 (0=opaque, 255=fully transparent). Default: 112.",
    )
    ap.add_argument(
        "--no-interp",
        action="store_true",
        help="Disable per-second work time and per-meter distance interpolation (smaller ASS output).",
    )
    ap.add_argument(
        "--lint",
        action="store_true",
        help="Lint the generated .ass output and exit non-zero on errors.",
    )
    ap.add_argument(
        "--lint-strict",
        action="store_true",
        help="Like --lint, but also fails on warnings.",
    )

    ap.add_argument(
        "--burn-in",
        metavar="OUT_VIDEO",
        default=None,
        help="If set, burn the overlay into a new video using ffmpeg.",
    )
    ap.add_argument(
        "--crf", type=int, default=18, help="x264 CRF for burn-in (default: 18)"
    )
    ap.add_argument(
        "--preset",
        default="veryfast",
        help="x264 preset for burn-in (default: veryfast)",
    )
    ap.add_argument(
        "--reencode-audio",
        action="store_true",
        help="Re-encode audio to AAC instead of stream-copying it (use if -c:a copy fails).",
    )

    ap.add_argument(
        "--ffprobe-bin", default="ffprobe", help="Path to ffprobe (default: ffprobe)"
    )
    ap.add_argument(
        "--ffmpeg-bin", default="ffmpeg", help="Path to ffmpeg (default: ffmpeg)"
    )
    ap.add_argument(
        "--anchor",
        dest="anchor",
        choices=["start", "first-visible", "first-row-visible"],
        default="start",
        help="Which sample to treat as time 0 for overlay generation (default: start).",
    )

    args = ap.parse_args()

    # Tools check
    if shutil.which(args.ffprobe_bin) is None:
        print(f"ERROR: ffprobe not found: {args.ffprobe_bin}", file=sys.stderr)
        return 2
    if args.burn_in and shutil.which(args.ffmpeg_bin) is None:
        print(f"ERROR: ffmpeg not found: {args.ffmpeg_bin}", file=sys.stderr)
        return 2

    video_path = args.video
    data_path = args.fit
    out_ass = args.out_ass or str(Path(video_path).with_suffix(".ass"))

    # Parse data (.fit)
    try:
        parsed = parse_data_file(data_path)
    except Exception as e:
        print(f"ERROR: Could not parse data file: {data_path}\n{e}", file=sys.stderr)
        return 2
    samples_all = parsed.samples
    if not samples_all:
        print(f"ERROR: No samples found in data file: {data_path}", file=sys.stderr)
        return 2

    data_start = samples_all[0].t

    # Probe video
    w, h, duration, video_creation_meta, source_meta = get_video_metadata(
        video_path, ffprobe_bin=args.ffprobe_bin
    )
    video_creation = video_creation_meta
    source_used = source_meta
    if args.video_start:
        try:
            video_creation = parse_iso8601(args.video_start)
        except Exception as e:
            print(f"ERROR: Could not parse --video-start: {args.video_start}\n{e}", file=sys.stderr)
            return 2
        source_used = "user_override"

    # Choose anchor (t0) used for both alignment and displayed elapsed time.
    anchor_idx = choose_anchor_index(
        samples_all, video_start=video_creation, mode=args.anchor
    )
    samples = samples_all[anchor_idx:] if anchor_idx else samples_all
    anchor_time = samples[0].t

    # Auto offset: when does anchor occur on the video timeline?
    auto_offset = (anchor_time - video_creation).total_seconds()
    offset = auto_offset + float(args.offset)

    print("== Alignment ==")
    print(f"Video creation/start time (UTC): {video_creation.isoformat()}  [{source_used}]")
    if args.video_start:
        print(
            f"Video metadata time (UTC):       {video_creation_meta.isoformat()}  [{source_meta}]"
        )
    if duration is not None:
        video_end = video_creation + timedelta(seconds=duration)
        print(
            f"Video end time (UTC):            {video_end.isoformat()}  [duration {duration:.2f} s]"
        )
    print(f"FIT file: {data_path}")
    print(f"FIT first timestamp (UTC):       {data_start.isoformat()}")
    delta0 = (data_start - video_creation).total_seconds()
    if abs(delta0) >= 1.0:
        when = "after" if delta0 > 0 else "before"
        print(
            f"FIT starts {abs(delta0):.1f} s {when} video start (based on absolute timestamps)."
        )
    first_row_visible = next(
        (s for s in samples_all if s.t >= video_creation and (s.cadence or 0) > 0), None
    )
    if first_row_visible is not None:
        tv = (first_row_visible.t - video_creation).total_seconds()
        print(
            f"First sample with cadence>0 during video: t={tv:.1f} s  [{first_row_visible.t.isoformat()}]"
        )
    if anchor_time != data_start:
        print(
            f"Data anchor ({args.anchor}) time (UTC): {anchor_time.isoformat()}  [idx {anchor_idx}]"
        )
    print(f"Auto offset (anchor - video_start): {auto_offset:+.3f} s")
    if args.offset:
        print(f"Manual adjustment: {args.offset:+.3f} s")
    print(f"Final offset used: {offset:+.3f} s")
    if duration is not None:
        print(f"Video: {w}x{h}, duration ~ {duration:.2f} s")
    else:
        print(f"Video: {w}x{h}, duration unknown")

    # Write ASS
    label_font = args.label_font
    value_font = args.value_font
    if args.font:
        label_font = args.font
        value_font = args.font
    try:
        generate_ass(
            samples=samples,
            out_ass=out_ass,
            video_w=w,
            video_h=h,
            video_duration=duration,
            offset_seconds=offset,
            label_font=label_font,
            value_font=value_font,
            value_fs=args.fontsize,
            left_margin=args.left_margin,
            top_margin=args.top_margin,
            bottom_margin=args.bottom_margin,
            box_alpha=args.box_alpha,
            interpolate=(not args.no_interp),
            laps=parsed.laps,
        )
    except Exception as e:
        print(f"ERROR: Could not generate ASS overlay:\n{e}", file=sys.stderr)
        return 2
    print(f"Wrote ASS overlay: {out_ass}")

    # Lint (optional)
    if args.lint or args.lint_strict:
        from c2_overlay.ass_lint import SEVERITY_RANK, lint_ass_file, print_issues

        issues = lint_ass_file(out_ass)
        if issues:
            print(f"== ASS Lint ({len(issues)} issue(s)) ==", file=sys.stderr)
            print_issues(issues)

        fail_on = "warn" if args.lint_strict else "error"
        fail_rank = SEVERITY_RANK[fail_on]
        if any(SEVERITY_RANK.get(i.severity, 0) >= fail_rank for i in issues):
            print(f"ERROR: ASS lint failed (fail-on {fail_on}).", file=sys.stderr)
            return 1

    # Burn-in (optional)
    if args.burn_in:
        print(f"Burning in subtitles to: {args.burn_in}")
        burn_in(
            video_in=video_path,
            ass_path=out_ass,
            video_out=args.burn_in,
            ffmpeg_bin=args.ffmpeg_bin,
            crf=args.crf,
            preset=args.preset,
            copy_audio=(not args.reencode_audio),
        )
        print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
