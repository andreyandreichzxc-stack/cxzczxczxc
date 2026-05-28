"""mcp_video tool — registered via @tool decorator.

Video analysis via ffprobe/ffmpeg subprocess calls.

Actions:
- ``action="info" path="data/video.mp4"`` — ffprobe format/stream data.
- ``action="frame" path="data/video.mp4" time="00:00:05"`` — extract single frame as JPEG.
- ``action="thumbnail" path="data/video.mp4"`` — extract thumbnail at 10% of duration.

Path must be within ``settings.data_dir``.  ffprobe/ffmpeg must be in PATH.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import settings
from src.core.actions.mcp_tools import _safe_resolve
from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_FRAME_DIR = "video_frames"  # sub-directory inside data_dir for extracted frames


# ── Helper: check ffmpeg availability ────────────────────────────────────


def _check_ffmpeg() -> str | None:
    """Return ``None`` if ffprobe and ffmpeg are available, or an error string."""
    for cmd in ("ffprobe", "ffmpeg"):
        try:
            subprocess.run(
                [cmd, "-version"],
                capture_output=True,
                check=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.CalledProcessError, TimeoutError):
            return "ffmpeg not installed or not in PATH"
    return None


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_video
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_video",
    description=(
        "Analyse videos via ffprobe/ffmpeg subprocess calls.\n\n"
        "Actions:\n"
        "- **info** — get duration, resolution, codec, bitrate, framerate via ffprobe.\n"
        "- **frame** — extract a single frame as JPEG at given time.\n"
        "- **thumbnail** — extract a thumbnail at 10%% of duration.\n\n"
        "Examples:\n"
        '  action="info" path="data/video.mp4"\n'
        '  action="frame" path="data/video.mp4" time="00:00:05"\n'
        '  action="thumbnail" path="data/video.mp4"'
    ),
    category="media",
    risk="low",
    params={
        "action": "str — 'info', 'frame', or 'thumbnail'",
        "path": "str — relative path inside data/ to the video file (required)",
        "time": "str — timestamp for frame extraction, e.g. '00:00:05' (default '00:00:01')",
    },
)
async def mcp_video(
    action: str = "",
    path: str = "",
    time: str = "00:00:01",
    **kwargs: Any,
) -> dict[str, Any]:
    """Video analysis tool via ffprobe/ffmpeg."""
    try:
        # Validate ffmpeg/ffprobe are available
        ffmpeg_err = _check_ffmpeg()
        if ffmpeg_err:
            return {"error": ffmpeg_err}

        if action not in ("info", "frame", "thumbnail"):
            return {
                "error": (
                    f"Unknown action {action!r}. Valid actions: info, frame, thumbnail"
                )
            }

        if not path or not path.strip():
            return {"error": "path parameter is required"}

        resolved = _safe_resolve(path.strip())
        if resolved is None:
            return {
                "error": (
                    f"Path {path!r} is outside allowed directories or contains '..'"
                )
            }
        if not resolved.is_file():
            return {"error": f"File not found: {resolved}"}

        if action == "info":
            return await _video_info(resolved)
        elif action == "frame":
            return await _extract_frame(resolved, time)
        else:  # thumbnail
            return await _extract_thumbnail(resolved)
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.exception("mcp_video(%r) failed", action)
        return {"error": f"Unexpected error: {exc}"}


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════


async def _video_info(video_path: Path) -> dict[str, Any]:
    """Get video metadata via ffprobe."""
    loop = asyncio.get_running_loop()

    def _probe() -> dict[str, Any]:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=30
            )
        except subprocess.TimeoutExpired:
            raise ValueError("ffprobe timed out")
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"ffprobe failed: {exc.stderr or exc.stdout or exc}")

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(f"ffprobe output is not valid JSON: {exc}")

        # Parse streams
        streams = data.get("streams", [])
        video_stream = None
        audio_streams = []
        for s in streams:
            if s.get("codec_type") == "video" and video_stream is None:
                video_stream = s
            elif s.get("codec_type") == "audio":
                audio_streams.append(s)

        info: dict[str, Any] = {
            "file_size_bytes": video_path.stat().st_size,
        }

        # Format info
        fmt = data.get("format", {})
        if fmt:
            info["format_name"] = fmt.get("format_name")
            duration_str = fmt.get("duration")
            if duration_str:
                try:
                    info["duration_seconds"] = round(float(duration_str), 2)
                except (ValueError, TypeError):
                    pass
            bitrate_str = fmt.get("bit_rate")
            if bitrate_str:
                try:
                    info["bitrate_bps"] = int(bitrate_str)
                except (ValueError, TypeError):
                    pass

        # Video stream info
        if video_stream:
            info["codec"] = video_stream.get("codec_name")
            info["width"] = video_stream.get("width")
            info["height"] = video_stream.get("height")
            w = video_stream.get("width")
            h = video_stream.get("height")
            if w and h:
                info["resolution"] = f"{w}x{h}"
            fps_str = video_stream.get("r_frame_rate")
            if fps_str and "/" in fps_str:
                try:
                    num, den = fps_str.split("/")
                    if int(den) > 0:
                        info["framerate"] = round(int(num) / int(den), 3)
                except (ValueError, ZeroDivisionError):
                    info["framerate"] = fps_str
            else:
                info["framerate"] = fps_str
            pix_fmt = video_stream.get("pix_fmt")
            if pix_fmt:
                info["pixel_format"] = pix_fmt
            info["video_stream_index"] = video_stream.get("index")

        # Audio info
        if audio_streams:
            info["audio_streams"] = len(audio_streams)
            info["audio_codec"] = audio_streams[0].get("codec_name")

        info["stream_count"] = len(streams)
        return info

    try:
        result = await loop.run_in_executor(None, _probe)
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.warning("Video info error: %s", exc)
        return {"error": f"Failed to probe video: {exc}"}

    return {"ok": True, **result}


async def _extract_frame(video_path: Path, time: str) -> dict[str, Any]:
    """Extract a single frame at *time* as JPEG."""
    loop = asyncio.get_running_loop()

    frames_dir = settings.data_dir / _FRAME_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"frame_{timestamp}.jpg"
    output_path = frames_dir / output_filename

    def _extract() -> dict[str, Any]:
        frames_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-ss",
            time,
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-q:v",
            "2",
            "-y",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        except subprocess.TimeoutExpired:
            raise ValueError("ffmpeg timed out during frame extraction")
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace")[:500]
            raise ValueError(f"ffmpeg failed: {stderr}")

        if not output_path.is_file():
            raise ValueError("ffmpeg did not produce an output file")

        file_size = output_path.stat().st_size
        return {
            "output": str(output_path),
            "filename": output_filename,
            "time": time,
            "size_bytes": file_size,
            "size_kb": round(file_size / 1024, 1),
        }

    try:
        result = await loop.run_in_executor(None, _extract)
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.warning("Frame extraction error: %s", exc)
        return {"error": f"Failed to extract frame: {exc}"}

    return {"ok": True, **result}


async def _extract_thumbnail(video_path: Path) -> dict[str, Any]:
    """Extract a thumbnail at 10% of video duration."""
    loop = asyncio.get_running_loop()

    frames_dir = settings.data_dir / _FRAME_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"thumbnail_{timestamp}.jpg"
    output_path = frames_dir / output_filename

    def _extract_thumb() -> dict[str, Any]:
        # 1. Get duration via ffprobe
        probe_cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            str(video_path),
        ]
        try:
            probe_result = subprocess.run(
                probe_cmd, capture_output=True, text=True, check=True, timeout=15
            )
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"ffprobe failed: {exc.stderr or exc}")

        try:
            probe_data = json.loads(probe_result.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(f"ffprobe output is not valid JSON: {exc}")

        fmt = probe_data.get("format", {})
        duration_str = fmt.get("duration", "0")
        try:
            duration = float(duration_str)
        except (ValueError, TypeError):
            raise ValueError("Could not determine video duration for thumbnail")

        thumb_time = duration * 0.1
        hours = int(thumb_time // 3600)
        minutes = int((thumb_time % 3600) // 60)
        seconds = int(thumb_time % 60)
        millis = int((thumb_time - int(thumb_time)) * 1000)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"

        # 2. Extract frame at calculated time
        frames_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-ss",
            time_str,
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-q:v",
            "2",
            "-y",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        except subprocess.TimeoutExpired:
            raise ValueError("ffmpeg timed out during thumbnail extraction")
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace")[:500]
            raise ValueError(f"ffmpeg failed: {stderr}")

        if not output_path.is_file():
            raise ValueError("ffmpeg did not produce an output file")

        file_size = output_path.stat().st_size
        return {
            "output": str(output_path),
            "filename": output_filename,
            "time": time_str,
            "duration_seconds": round(duration, 2),
            "thumbnail_at_percent": 10,
            "size_bytes": file_size,
            "size_kb": round(file_size / 1024, 1),
        }

    try:
        result = await loop.run_in_executor(None, _extract_thumb)
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.warning("Thumbnail extraction error: %s", exc)
        return {"error": f"Failed to extract thumbnail: {exc}"}

    return {"ok": True, **result}
