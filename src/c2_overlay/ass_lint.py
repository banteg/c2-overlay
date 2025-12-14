from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


TIME_RE = re.compile(r"^(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2})\.(?P<cs>\d{2})$")
POS_RE = re.compile(
    r"\\pos\(\s*(?P<x>-?\d+(?:\.\d+)?)\s*,\s*(?P<y>-?\d+(?:\.\d+)?)\s*\)"
)
TAG_RE = re.compile(r"{[^}]*}")


def parse_ass_time(s: str) -> float | None:
    m = TIME_RE.match(s.strip())
    if not m:
        return None
    h = int(m.group("h"))
    mi = int(m.group("m"))
    sec = int(m.group("s"))
    cs = int(m.group("cs"))
    return float(h * 3600 + mi * 60 + sec) + (cs / 100.0)


def strip_ass_tags(text: str) -> str:
    return TAG_RE.sub("", text).strip()


def parse_pos(text: str) -> tuple[int, int] | None:
    m = POS_RE.search(text)
    if not m:
        return None
    x = int(float(m.group("x")))
    y = int(float(m.group("y")))
    return x, y


@dataclass(frozen=True)
class AssEvent:
    line_no: int
    layer: int
    start: float
    end: float
    style: str
    text: str
    plain: str
    pos: tuple[int, int] | None
    is_drawing: bool


@dataclass(frozen=True)
class LintIssue:
    code: str
    severity: str  # "error" | "warn" | "info"
    message: str
    line_no: int | None = None
    related: tuple[int, ...] = ()


SEVERITY_RANK = {"info": 0, "warn": 1, "error": 2}


def parse_ass_events(text: str) -> tuple[list[AssEvent], list[LintIssue]]:
    issues: list[LintIssue] = []
    events: list[AssEvent] = []

    in_events = False
    format_cols: list[str] | None = None
    for idx, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.lower() == "[events]":
            in_events = True
            continue
        if not in_events:
            continue
        if line.lower().startswith("format:"):
            cols = line.split(":", 1)[1].split(",")
            format_cols = [c.strip() for c in cols if c.strip()]
            continue
        if not line.lower().startswith("dialogue:"):
            continue

        if format_cols is None:
            format_cols = [
                "Layer",
                "Start",
                "End",
                "Style",
                "Name",
                "MarginL",
                "MarginR",
                "MarginV",
                "Effect",
                "Text",
            ]

        payload = raw.split(":", 1)[1].lstrip()
        parts = payload.split(",", len(format_cols) - 1)
        if len(parts) != len(format_cols):
            issues.append(
                LintIssue(
                    code="ASS001",
                    severity="error",
                    message=f"Dialogue field count mismatch: expected {len(format_cols)}, got {len(parts)}",
                    line_no=idx,
                )
            )
            continue

        fields = dict(zip(format_cols, parts))
        try:
            layer = int(fields.get("Layer", "0").strip())
        except ValueError:
            issues.append(
                LintIssue(
                    code="ASS001",
                    severity="error",
                    message=f"Invalid Layer value: {fields.get('Layer')!r}",
                    line_no=idx,
                )
            )
            continue

        start_s = parse_ass_time(fields.get("Start", "").strip())
        end_s = parse_ass_time(fields.get("End", "").strip())
        if start_s is None or end_s is None:
            issues.append(
                LintIssue(
                    code="ASS001",
                    severity="error",
                    message=f"Invalid Start/End time: {fields.get('Start')!r} .. {fields.get('End')!r}",
                    line_no=idx,
                )
            )
            continue

        style = fields.get("Style", "").strip()
        text_field = fields.get("Text", "")
        plain = strip_ass_tags(text_field)
        pos = parse_pos(text_field)
        is_drawing = bool(re.search(r"\\p([1-9]\d*)", text_field))

        events.append(
            AssEvent(
                line_no=idx,
                layer=layer,
                start=start_s,
                end=end_s,
                style=style,
                text=text_field,
                plain=plain,
                pos=pos,
                is_drawing=is_drawing,
            )
        )

    return events, issues


def lint_ass_events(
    events: list[AssEvent],
    *,
    max_overlap_issues: int = 50,
    ignore_styles: set[str] | None = None,
    require_pos_styles: set[str] | None = None,
) -> list[LintIssue]:
    ignore_styles = ignore_styles or {"Box"}
    require_pos_styles = require_pos_styles or {
        "Time",
        "Split",
        "SPM",
        "Distance",
        "Watts",
        "HeartRate",
        "Label",
    }

    issues: list[LintIssue] = []

    # Basic time sanity
    for ev in events:
        if ev.end <= ev.start:
            issues.append(
                LintIssue(
                    code="ASS010",
                    severity="error",
                    message=f"Non-positive duration event ({ev.start:.2f}s -> {ev.end:.2f}s) style={ev.style}",
                    line_no=ev.line_no,
                )
            )
        if ev.start < 0 or ev.end < 0:
            issues.append(
                LintIssue(
                    code="ASS011",
                    severity="error",
                    message=f"Negative timestamp ({ev.start:.2f}s -> {ev.end:.2f}s) style={ev.style}",
                    line_no=ev.line_no,
                )
            )
        if ev.style in require_pos_styles and not ev.is_drawing and ev.pos is None:
            issues.append(
                LintIssue(
                    code="ASS012",
                    severity="warn",
                    message=f"Missing \\pos() tag for style={ev.style}",
                    line_no=ev.line_no,
                )
            )

    # Overdraw / z-fighting detection: overlapping events for same style+pos with different visible text.
    groups: dict[tuple[str, tuple[int, int] | None], list[AssEvent]] = {}
    for ev in events:
        if ev.style in ignore_styles:
            continue
        if ev.is_drawing:
            continue
        key = (ev.style, ev.pos)
        groups.setdefault(key, []).append(ev)

    eps = 1e-6
    reported_pairs: set[tuple[int, int]] = set()
    overlap_issues = 0
    for (style, pos), group in groups.items():
        group = sorted(group, key=lambda e: (e.start, e.end, e.line_no))
        active: list[AssEvent] = []
        for ev in group:
            active = [a for a in active if a.end > ev.start + eps]
            for a in active:
                if a.plain == ev.plain:
                    continue
                pair = (min(a.line_no, ev.line_no), max(a.line_no, ev.line_no))
                if pair in reported_pairs:
                    continue
                reported_pairs.add(pair)
                issues.append(
                    LintIssue(
                        code="ASS020",
                        severity="error",
                        message=(
                            f"Overlapping draw for style={style} pos={pos}: "
                            f"'{a.plain}' overlaps '{ev.plain}'"
                        ),
                        line_no=ev.line_no,
                        related=(a.line_no,),
                    )
                )
                overlap_issues += 1
                if overlap_issues >= max_overlap_issues:
                    issues.append(
                        LintIssue(
                            code="ASS020",
                            severity="warn",
                            message=f"Too many overlap issues; stopping at {max_overlap_issues}",
                        )
                    )
                    return issues
            active.append(ev)

    # File size / density heuristic
    if len(events) > 200_000:
        issues.append(
            LintIssue(
                code="ASS030",
                severity="warn",
                message=f"Very large ASS: {len(events)} Dialogue events (rendering may be slow)",
            )
        )

    return issues


def lint_ass_text(text: str) -> list[LintIssue]:
    events, parse_issues = parse_ass_events(text)
    issues = parse_issues + lint_ass_events(events)
    issues.sort(
        key=lambda i: (-SEVERITY_RANK.get(i.severity, 0), i.line_no or 0, i.code)
    )
    return issues


def lint_ass_file(path: str | Path) -> list[LintIssue]:
    p = Path(path)
    return lint_ass_text(p.read_text(encoding="utf-8"))


def issues_to_json(issues: list[LintIssue]) -> str:
    return json.dumps(
        [
            {
                "code": i.code,
                "severity": i.severity,
                "message": i.message,
                "line": i.line_no,
                "related": list(i.related),
            }
            for i in issues
        ],
        indent=2,
    )


def print_issues(issues: list[LintIssue]) -> None:
    for i in issues:
        loc = f":{i.line_no}" if i.line_no is not None else ""
        rel = f" (related: {', '.join(map(str, i.related))})" if i.related else ""
        print(f"{i.severity.upper()} {i.code}{loc}: {i.message}{rel}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="c2-overlay-lint", description="Lint ASS subtitle files.")
    ap.add_argument("ass", nargs="+", help="ASS file(s) to lint")
    ap.add_argument("--json", action="store_true", help="Output issues as JSON")
    ap.add_argument(
        "--fail-on",
        choices=["info", "warn", "error"],
        default="error",
        help="Exit non-zero when issues at/above this severity exist (default: error)",
    )
    args = ap.parse_args(argv)

    fail_rank = SEVERITY_RANK[args.fail_on]
    all_issues: list[LintIssue] = []
    for path in args.ass:
        issues = lint_ass_file(path)
        if issues:
            header = LintIssue(
                code="ASS000",
                severity="info",
                message=f"File: {path} ({len(issues)} issue(s))",
            )
            all_issues.append(header)
            all_issues.extend(issues)

    if args.json:
        print(issues_to_json(all_issues))
    else:
        print_issues(all_issues)

    if any(SEVERITY_RANK.get(i.severity, 0) >= fail_rank for i in all_issues):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
