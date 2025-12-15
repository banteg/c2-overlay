# Concept2 timestamp precision caveat (video sync)

When syncing Concept2 stroke data to external media (e.g. aligning strokes to a workout video), **do not assume Concept2 “workout timestamp” is a precise start time**.

Two common gotchas:

1. **Concept2 Logbook API timestamp is end-of-workout, not start.**
   If you use the Concept2 Online Logbook API, the `date` field corresponds to the time **as stored in the monitor**, and is associated with the **end of the workout**, not the beginning. Any “start time” computed from this must be derived as `start ≈ end − duration` (still approximate).

2. **Clock metadata can be coarse / truncated (minute-level).**
   Depending on the export route, some Concept2 summaries omit seconds (HH:MM). When aligning to a video’s wall-clock timestamp (e.g. `creation_time`), this introduces up to **±59 seconds of unavoidable ambiguity**, even if everything else is correct.

## Recommendation for implementers

Treat Concept2 “absolute time” as a **rough initial guess**, not ground truth:

- Use per-stroke/elapsed-time as the primary timeline.
- Provide explicit correction knobs.

In `c2-overlay`, the knobs are:

- `--offset SECONDS`: manual adjustment applied to the alignment.
- `--video-start ISO8601`: override the video start timestamp when metadata is missing/incorrect.
- `--workout-start SECONDS`: override auto alignment and place the selected anchor sample at a specific time in the video.

## Suggested user-facing note

> **Note on timestamp accuracy:** Concept2 workout time metadata is not guaranteed to be start-aligned or second-precise. Some exports omit seconds entirely, effectively truncating time-of-day to the minute. As a result, video synchronization based on wall-clock time may be off by up to ~1 minute and should be corrected using a manual/automatic offset.

