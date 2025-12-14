# c2-overlay

<img src="docs/hero.png" width="720" />

Generate PM5-style overlays for rowing videos from Concept2 FIT files.

Creates accurate ASS subtitle overlays matching the Concept2 PM5 monitor display, synchronized with your workout video using metadata timestamps.

## Features

- **PM5-style display**: Shows time, split pace, stroke rate, distance, watts, and heart rate in a familiar layout
- **Automatic synchronization**: Aligns workout data with video using creation timestamps
- **Interval workout support**: Displays work/rest lap information with visual indicators
- **Rest overlay**: Shows previous lap statistics during rest periods
- **Flexible output**: Generate ASS subtitles or burn them directly into video

## Installation

```bash
uvx c2-overlay video.mp4 workout.fit
```

Or install locally:

```bash
uv tool install c2-overlay
```

## Usage

Basic usage (generates ASS subtitle file):

```bash
uvx c2-overlay video.mp4 workout.fit
```

Generate and lint the ASS output:

```bash
uvx c2-overlay video.mp4 workout.fit --lint
```

Lint an existing ASS file:

```bash
uvx c2-overlay-lint video.ass
```

Burn overlay into video:

```bash
uvx c2-overlay video.mp4 workout.fit --burn-in output.mp4
```

Adjust timing manually:

```bash
uvx c2-overlay video.mp4 workout.fit --offset 2.5
```

Override the video start timestamp (when metadata is missing/incorrect):

```bash
uvx c2-overlay video.mp4 workout.fit --video-start 2025-12-14T10:41:31Z
```

### Options

- `-o, --out-ass PATH`: Output .ass file path (default: same name as video)
- `--video-start ISO8601`: Override video start timestamp (e.g. `2025-12-14T10:41:31Z`)
- `--offset SECONDS`: Manual offset adjustment in seconds (positive = later, negative = earlier)
- `--burn-in OUT_VIDEO`: Burn overlay into video using ffmpeg
- `--label-font FONT`: Font for labels (default: PragmataPro)
- `--value-font FONT`: Font for values (default: PragmataPro Mono)
- `--fontsize SIZE`: Value font size (default: scaled from resolution)
- `--left-margin PX`: Left margin in pixels
- `--top-margin PX`: Top margin (positions overlay from top)
- `--bottom-margin PX`: Bottom margin (default positioning)
- `--box-alpha 0-255`: Background transparency (0=opaque, 255=transparent, default: 112)
- `--no-interp`: Disable per-second/per-meter interpolation (smaller ASS output)
- `--lint`: Lint the generated ASS output and exit non-zero on errors
- `--lint-strict`: Like `--lint`, but also fails on warnings
- `--crf VALUE`: Video quality for burn-in (default: 18)
- `--preset PRESET`: Encoding preset (default: veryfast)

## Requirements

- Python 3.12+
- ffprobe (for video metadata)
- ffmpeg (optional, for burn-in)

## How it Works

1. Extracts video creation timestamp from metadata
2. Reads FIT file trackpoints with absolute timestamps
3. Computes synchronization offset
4. Generates ASS subtitle file with PM5-style overlay
5. Optionally burns subtitles into video with ffmpeg

## Display Layout

The overlay shows a bottom-left panel with two rows:

**Row 1**: TIME | SPLIT | S/M
**Row 2**: METERS | WATTS | BPM

During rest intervals, the overlay displays:
- Previous lap statistics
- Rest countdown timer
- Visual tinting and border highlighting

## License

MIT
