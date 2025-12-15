from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fitparse import FitFile

from .timeutil import to_utc


@dataclass
class Sample:
    t: datetime
    distance_m: float | None = None
    hr: int | None = None
    cadence: int | None = None  # usually stroke rate (SPM) in Concept2 FIT
    watts: int | None = None
    speed: float | None = None  # m/s


@dataclass(frozen=True)
class LapSegment:
    index: int  # 1-based
    start: datetime
    end: datetime
    intensity: str  # "active" | "rest" | other
    start_distance_m: float | None
    total_elapsed_s: float | None
    total_distance_m: float | None
    avg_speed_m_s: float | None
    avg_power_w: int | None
    avg_cadence_spm: int | None
    avg_hr_bpm: int | None


INTENSITY_MAP = {0: "active", 1: "rest"}


def normalize_intensity(v: object) -> str:
    if v is None:
        return "unknown"
    if isinstance(v, (int, float)):
        return INTENSITY_MAP.get(int(v), f"unknown({int(v)})")
    s = str(v).strip().lower()
    if s in {"active", "rest"}:
        return s
    return s or "unknown"


def parse_fit_messages(fit: FitFile) -> list[Sample]:
    samples: list[Sample] = []
    for msg in fit.get_messages("record"):
        fields = {f.name: f.value for f in msg}
        ts = fields.get("timestamp")
        if not isinstance(ts, datetime):
            continue
        ts = to_utc(ts)

        distance_m = fields.get("distance")
        if isinstance(distance_m, (int, float)):
            distance_m = float(distance_m)
        else:
            distance_m = None

        hr = fields.get("heart_rate")
        hr = int(hr) if isinstance(hr, (int, float)) else None

        cadence = fields.get("cadence")
        cadence = int(cadence) if isinstance(cadence, (int, float)) else None

        watts = fields.get("power")
        watts = int(watts) if isinstance(watts, (int, float)) else None

        speed = fields.get("enhanced_speed", fields.get("speed"))
        speed = float(speed) if isinstance(speed, (int, float)) else None

        samples.append(
            Sample(
                t=ts,
                distance_m=distance_m,
                hr=hr,
                cadence=cadence,
                watts=watts,
                speed=speed,
            )
        )

    samples.sort(key=lambda s: s.t)

    # Fill in missing speeds from distance/time deltas (m/s)
    for i in range(1, len(samples)):
        cur = samples[i]
        prev = samples[i - 1]
        d_cur = cur.distance_m
        d_prev = prev.distance_m
        if cur.speed is None and d_cur is not None and d_prev is not None:
            dt = (cur.t - prev.t).total_seconds()
            dd = d_cur - d_prev
            if dt > 0 and dd >= 0:
                cur.speed = dd / dt
    if samples and samples[0].speed is None and len(samples) > 1:
        samples[0].speed = samples[1].speed

    return samples


@dataclass(frozen=True)
class ParsedData:
    samples: list[Sample]
    laps: list[LapSegment] | None = None


def parse_data_file(path: str) -> ParsedData:
    ext = Path(path).suffix.lower()
    if ext == ".fit":
        fit = FitFile(path)
        samples = parse_fit_messages(fit)
        laps: list[LapSegment] = []
        # Build a timestamp list for fast lookup of lap start distances.
        ts_list = [s.t for s in samples]
        dist_list = [s.distance_m for s in samples]

        def distance_at(t: datetime) -> float | None:
            idx = min(bisect_left(ts_list, t), len(ts_list) - 1)
            # Prefer the first sample at/after start; fallback to previous if missing distance.
            for j in (idx, idx - 1, idx + 1):
                if 0 <= j < len(dist_list):
                    d = dist_list[j]
                    if d is not None:
                        return float(d)
            return None

        for msg in fit.get_messages("lap"):
            fields = {f.name: f.value for f in msg}
            start = fields.get("start_time")
            end = fields.get("timestamp")
            if not isinstance(start, datetime) or not isinstance(end, datetime):
                continue
            start = to_utc(start)
            end = to_utc(end)

            total_elapsed = fields.get("total_elapsed_time")
            total_distance = fields.get("total_distance")
            avg_speed = fields.get("enhanced_avg_speed", fields.get("avg_speed"))
            avg_power = fields.get("avg_power")
            avg_cadence = fields.get("avg_cadence")
            avg_hr = fields.get("avg_heart_rate")

            laps.append(
                LapSegment(
                    index=int(fields.get("message_index", len(laps))) + 1,
                    start=start,
                    end=end,
                    intensity=normalize_intensity(fields.get("intensity")),
                    start_distance_m=distance_at(start),
                    total_elapsed_s=float(total_elapsed)
                    if isinstance(total_elapsed, (int, float))
                    else None,
                    total_distance_m=float(total_distance)
                    if isinstance(total_distance, (int, float))
                    else None,
                    avg_speed_m_s=float(avg_speed)
                    if isinstance(avg_speed, (int, float))
                    else None,
                    avg_power_w=int(avg_power)
                    if isinstance(avg_power, (int, float))
                    else None,
                    avg_cadence_spm=int(avg_cadence)
                    if isinstance(avg_cadence, (int, float))
                    else None,
                    avg_hr_bpm=int(avg_hr)
                    if isinstance(avg_hr, (int, float))
                    else None,
                )
            )

        laps.sort(key=lambda lap: lap.start)
        return ParsedData(samples=samples, laps=laps or None)

    raise ValueError(f"Unsupported data file extension: {ext} (expected .fit)")
