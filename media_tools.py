"""
media_tools.py — Image and video processing tools for the media specialist agent.

Dependencies: Pillow, moviepy, opencv-python
Install: pip install Pillow moviepy opencv-python
"""

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_file(path: str) -> str | None:
    """Returns an error message if path doesn't exist, else None."""
    if not Path(path).exists():
        return f"File not found: '{path}'"
    return None


def _output_path(input_path: str, suffix: str, ext: str) -> str:
    """Builds a default output path next to the input file."""
    p = Path(input_path)
    return str(p.parent / f"{p.stem}{suffix}{ext}")


# ---------------------------------------------------------------------------
# IMAGE TOOLS
# ---------------------------------------------------------------------------

def get_image_info(image_path: str) -> dict:
    """
    Returns technical information about an image file: dimensions, format,
    color mode, file size, and EXIF metadata (camera, GPS, date taken) if available.

    Args:
        image_path: Full path to the image file (JPEG, PNG, WEBP, TIFF, etc.).
    """
    err = _check_file(image_path)
    if err:
        return {"status": "error", "error_message": err}

    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        with Image.open(image_path) as img:
            file_size_kb = round(os.path.getsize(image_path) / 1024, 1)

            # EXIF data
            exif_data = {}
            try:
                raw_exif = img._getexif()
                if raw_exif:
                    for tag_id, val in raw_exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if tag in ("Make", "Model", "DateTime", "GPSInfo",
                                   "ExposureTime", "FNumber", "ISOSpeedRatings",
                                   "FocalLength", "Flash", "Software"):
                            exif_data[tag] = str(val)
            except Exception:
                pass

            return {
                "status":      "success",
                "path":        image_path,
                "format":      img.format,
                "mode":        img.mode,
                "width_px":    img.width,
                "height_px":   img.height,
                "megapixels":  round(img.width * img.height / 1_000_000, 2),
                "file_size_kb": file_size_kb,
                "exif":        exif_data or "No EXIF data found",
            }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def edit_image(
    image_path: str,
    output_path: str = "",
    resize_width: int = 0,
    resize_height: int = 0,
    crop_box: str = "",
    rotate_degrees: float = 0.0,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    convert_format: str = "",
    grayscale: bool = False,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> dict:
    """
    Edit an image: resize, crop, rotate, adjust brightness/contrast/saturation,
    convert format, apply grayscale, or flip. Saves the result to output_path.

    Args:
        image_path:      Path to the source image.
        output_path:     Where to save the result. Defaults to <name>_edited.<ext>.
        resize_width:    Target width in pixels (0 = keep aspect ratio from height).
        resize_height:   Target height in pixels (0 = keep aspect ratio from width).
        crop_box:        Crop region as 'left,top,right,bottom' in pixels. E.g. '0,0,800,600'.
        rotate_degrees:  Degrees to rotate counter-clockwise (e.g. 90, 180, 270).
        brightness:      Brightness multiplier. 1.0 = original, 1.5 = brighter, 0.5 = darker.
        contrast:        Contrast multiplier. 1.0 = original, >1 = more contrast.
        saturation:      Saturation multiplier. 1.0 = original, 0 = grayscale, 2 = vivid.
        convert_format:  Output format: 'JPEG', 'PNG', 'WEBP', 'TIFF', 'BMP'.
        grayscale:       Convert to black and white if True.
        flip_horizontal: Mirror image left-to-right.
        flip_vertical:   Flip image upside-down.
    """
    err = _check_file(image_path)
    if err:
        return {"status": "error", "error_message": err}

    try:
        from PIL import Image, ImageEnhance

        with Image.open(image_path) as img:
            orig_size = (img.width, img.height)

            # Convert to RGB if needed for operations
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Grayscale
            if grayscale:
                img = img.convert("L").convert("RGB")

            # Crop
            if crop_box:
                parts = [int(x.strip()) for x in crop_box.split(",")]
                if len(parts) == 4:
                    img = img.crop(tuple(parts))

            # Resize
            if resize_width > 0 or resize_height > 0:
                w = resize_width or int(img.width * resize_height / img.height)
                h = resize_height or int(img.height * resize_width / img.width)
                img = img.resize((w, h), Image.LANCZOS)

            # Rotate
            if rotate_degrees != 0.0:
                img = img.rotate(rotate_degrees, expand=True)

            # Flip
            if flip_horizontal:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if flip_vertical:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

            # Brightness / Contrast / Saturation
            if brightness != 1.0:
                img = ImageEnhance.Brightness(img).enhance(brightness)
            if contrast != 1.0:
                img = ImageEnhance.Contrast(img).enhance(contrast)
            if saturation != 1.0:
                img = ImageEnhance.Color(img).enhance(saturation)

            # Output format
            fmt = convert_format.upper() if convert_format else (img.format or "JPEG")
            ext_map = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp",
                       "TIFF": ".tiff", "BMP": ".bmp"}
            ext = ext_map.get(fmt, ".jpg")

            out = output_path or _output_path(image_path, "_edited", ext)
            img.save(out, format=fmt)

            return {
                "status":       "success",
                "output_path":  out,
                "original_size": f"{orig_size[0]}x{orig_size[1]}",
                "new_size":     f"{img.width}x{img.height}",
                "format":       fmt,
                "operations":   [
                    op for op, applied in [
                        ("resize",           resize_width > 0 or resize_height > 0),
                        ("crop",             bool(crop_box)),
                        ("rotate",           rotate_degrees != 0.0),
                        ("flip_horizontal",  flip_horizontal),
                        ("flip_vertical",    flip_vertical),
                        ("brightness",       brightness != 1.0),
                        ("contrast",         contrast != 1.0),
                        ("saturation",       saturation != 1.0),
                        ("grayscale",        grayscale),
                        ("format_convert",   bool(convert_format)),
                    ] if applied
                ],
            }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def add_text_to_image(
    image_path: str,
    text: str,
    output_path: str = "",
    position: str = "bottom-center",
    font_size: int = 48,
    color: str = "white",
    background: bool = True,
) -> dict:
    """
    Overlays text onto an image. Useful for captions, watermarks, or labels.

    Args:
        image_path:  Path to the source image.
        text:        The text to overlay.
        output_path: Where to save the result. Defaults to <name>_text.<ext>.
        position:    Text placement: 'top-left', 'top-center', 'top-right',
                     'bottom-left', 'bottom-center', 'bottom-right', 'center'.
        font_size:   Font size in pixels (default 48).
        color:       Text color name or hex (e.g. 'white', 'black', '#FF0000').
        background:  If True, adds a semi-transparent dark box behind the text.
    """
    err = _check_file(image_path)
    if err:
        return {"status": "error", "error_message": err}

    try:
        from PIL import Image, ImageDraw, ImageFont

        with Image.open(image_path) as img:
            if img.mode != "RGBA":
                img = img.convert("RGBA")

            draw = ImageDraw.Draw(img)

            # Try to load a decent font, fall back to default
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except Exception:
                    font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            margin = 20

            # Calculate position
            iw, ih = img.size
            pos_map = {
                "top-left":      (margin, margin),
                "top-center":    ((iw - tw) // 2, margin),
                "top-right":     (iw - tw - margin, margin),
                "center":        ((iw - tw) // 2, (ih - th) // 2),
                "bottom-left":   (margin, ih - th - margin),
                "bottom-center": ((iw - tw) // 2, ih - th - margin),
                "bottom-right":  (iw - tw - margin, ih - th - margin),
            }
            x, y = pos_map.get(position, pos_map["bottom-center"])

            # Semi-transparent background box
            if background:
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                box_draw = ImageDraw.Draw(overlay)
                pad = 10
                box_draw.rectangle(
                    [x - pad, y - pad, x + tw + pad, y + th + pad],
                    fill=(0, 0, 0, 140),
                )
                img = Image.alpha_composite(img, overlay)
                draw = ImageDraw.Draw(img)

            draw.text((x, y), text, font=font, fill=color)
            img = img.convert("RGB")

            out = output_path or _output_path(image_path, "_text", ".jpg")
            img.save(out, format="JPEG")

            return {
                "status":      "success",
                "output_path": out,
                "text":        text,
                "position":    position,
            }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


# ---------------------------------------------------------------------------
# VIDEO TOOLS
# ---------------------------------------------------------------------------

def photos_to_video(
    image_paths: list,
    output_path: str = "",
    duration_per_image: float = 3.0,
    fps: int = 24,
    transition: str = "none",
    audio_path: str = "",
) -> dict:
    """
    Creates a video slideshow from a list of image files.
    Images are shown in order, each displayed for a set duration.

    Args:
        image_paths:         List of image file paths in display order.
        output_path:         Where to save the video (MP4). Defaults to slideshow.mp4 in first image's folder.
        duration_per_image:  How many seconds each image is shown (default 3.0).
        fps:                 Frames per second for the output video (default 24).
        transition:          Transition effect between images: 'none' or 'crossfade'.
        audio_path:          Optional path to an audio file (MP3/WAV) to add as background music.
    """
    if not image_paths:
        return {"status": "error", "error_message": "No image paths provided."}

    missing = [p for p in image_paths if not Path(p).exists()]
    if missing:
        return {"status": "error", "error_message": f"Files not found: {missing}"}

    try:
        from moviepy import ImageClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip

        clips = []
        for img_path in image_paths:
            clip = ImageClip(img_path, duration=duration_per_image)
            clips.append(clip)

        if transition == "crossfade" and len(clips) > 1:
            fade_dur = min(0.5, duration_per_image / 4)
            faded = [clips[0]]
            for c in clips[1:]:
                faded.append(c.with_start(faded[-1].end - fade_dur).crossfadein(fade_dur))
            final = CompositeAudioClip(faded) if audio_path else concatenate_videoclips(faded, method="compose")
        else:
            final = concatenate_videoclips(clips, method="compose")

        # Add audio if provided
        if audio_path and Path(audio_path).exists():
            audio = AudioFileClip(audio_path)
            if audio.duration > final.duration:
                audio = audio.subclipped(0, final.duration)
            final = final.with_audio(audio)

        out = output_path or str(Path(image_paths[0]).parent / "slideshow.mp4")
        final.write_videofile(out, fps=fps, logger=None)

        return {
            "status":         "success",
            "output_path":    out,
            "image_count":    len(image_paths),
            "total_duration": f"{len(image_paths) * duration_per_image:.1f}s",
            "fps":            fps,
            "transition":     transition,
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def get_video_info(video_path: str) -> dict:
    """
    Returns technical information about a video file: duration, resolution,
    frame rate, codec, audio tracks, and file size.

    Args:
        video_path: Full path to the video file (MP4, MOV, AVI, MKV, etc.).
    """
    err = _check_file(video_path)
    if err:
        return {"status": "error", "error_message": err}

    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "error", "error_message": f"Could not open video: {video_path}"}

        fps        = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = frame_count / fps if fps > 0 else 0
        cap.release()

        file_size_mb = round(os.path.getsize(video_path) / (1024 * 1024), 2)

        return {
            "status":        "success",
            "path":          video_path,
            "resolution":    f"{width}x{height}",
            "fps":           round(fps, 2),
            "frame_count":   frame_count,
            "duration":      f"{int(duration_s // 60)}m {int(duration_s % 60)}s",
            "duration_sec":  round(duration_s, 2),
            "file_size_mb":  file_size_mb,
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def edit_video(
    video_path: str,
    output_path: str = "",
    trim_start: float = 0.0,
    trim_end: float = 0.0,
    speed: float = 1.0,
    reverse: bool = False,
    mute: bool = False,
) -> dict:
    """
    Edit a video: trim, change speed, reverse playback, or mute audio.

    Args:
        video_path:   Path to the source video.
        output_path:  Where to save the result. Defaults to <name>_edited.mp4.
        trim_start:   Start time in seconds (0 = beginning).
        trim_end:     End time in seconds (0 = keep until end of video).
        speed:        Playback speed multiplier. 1.0 = normal, 2.0 = double speed, 0.5 = half speed.
        reverse:      Reverse the video playback direction.
        mute:         Remove all audio from the output.
    """
    err = _check_file(video_path)
    if err:
        return {"status": "error", "error_message": err}

    try:
        from moviepy import VideoFileClip

        clip = VideoFileClip(video_path)
        original_duration = clip.duration

        # Trim
        start = trim_start if trim_start > 0 else 0
        end   = trim_end if trim_end > 0 else clip.duration
        if start > 0 or end < clip.duration:
            clip = clip.subclipped(start, end)

        # Speed
        if speed != 1.0:
            clip = clip.with_effects([lambda c: c.multiply_speed(speed)])

        # Reverse
        if reverse:
            clip = clip.with_effects([lambda c: c.time_mirror()])

        # Mute
        if mute:
            clip = clip.without_audio()

        out = output_path or _output_path(video_path, "_edited", ".mp4")
        clip.write_videofile(out, logger=None)
        clip.close()

        return {
            "status":             "success",
            "output_path":        out,
            "original_duration":  f"{original_duration:.1f}s",
            "new_duration":       f"{clip.duration:.1f}s",
            "speed":              speed,
            "reversed":           reverse,
            "muted":              mute,
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def extract_video_frames(
    video_path: str,
    output_dir: str = "",
    interval_seconds: float = 1.0,
    max_frames: int = 50,
    format: str = "jpg",
) -> dict:
    """
    Extracts frames from a video at regular time intervals and saves them as images.
    Useful for reviewing video content, creating thumbnails, or preparing training data.

    Args:
        video_path:        Path to the source video.
        output_dir:        Directory to save extracted frames. Defaults to <video_name>_frames/.
        interval_seconds:  Extract one frame every N seconds (default 1.0).
        max_frames:        Maximum number of frames to extract (default 50).
        format:            Image format for saved frames: 'jpg' or 'png'.
    """
    err = _check_file(video_path)
    if err:
        return {"status": "error", "error_message": err}

    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "error", "error_message": f"Could not open video: {video_path}"}

        fps         = cap.get(cv2.CAP_PROP_FPS)
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval_frames = max(1, int(fps * interval_seconds))

        out_dir = Path(output_dir) if output_dir else Path(video_path).parent / f"{Path(video_path).stem}_frames"
        out_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        frame_idx   = 0
        saved_count = 0

        while saved_count < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            filename = out_dir / f"frame_{saved_count:04d}_t{frame_idx / fps:.1f}s.{format}"
            cv2.imwrite(str(filename), frame)
            saved_paths.append(str(filename))

            frame_idx   += interval_frames
            saved_count += 1

            if frame_idx >= frame_total:
                break

        cap.release()

        return {
            "status":          "success",
            "output_dir":      str(out_dir),
            "frames_extracted": saved_count,
            "interval_seconds": interval_seconds,
            "frame_paths":     saved_paths[:10],  # first 10 paths shown
            "note":            f"All {saved_count} frames saved to {out_dir}",
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def merge_videos(
    video_paths: list,
    output_path: str = "",
    transition: str = "none",
) -> dict:
    """
    Merges multiple video files into one by concatenating them in order.

    Args:
        video_paths:  List of video file paths to merge, in the desired order.
        output_path:  Where to save the merged video. Defaults to merged.mp4 in the first video's folder.
        transition:   Transition between clips: 'none' or 'crossfade'.
    """
    if not video_paths or len(video_paths) < 2:
        return {"status": "error", "error_message": "Provide at least 2 video paths to merge."}

    missing = [p for p in video_paths if not Path(p).exists()]
    if missing:
        return {"status": "error", "error_message": f"Files not found: {missing}"}

    try:
        from moviepy import VideoFileClip, concatenate_videoclips

        clips = [VideoFileClip(p) for p in video_paths]

        if transition == "crossfade" and len(clips) > 1:
            fade_dur = 0.5
            for i in range(1, len(clips)):
                clips[i] = clips[i].with_start(clips[i - 1].end - fade_dur).crossfadein(fade_dur)
            final = concatenate_videoclips(clips, method="compose")
        else:
            final = concatenate_videoclips(clips, method="compose")

        out = output_path or str(Path(video_paths[0]).parent / "merged.mp4")
        final.write_videofile(out, logger=None)

        for c in clips:
            c.close()

        total_duration = sum(c.duration for c in clips)

        return {
            "status":         "success",
            "output_path":    out,
            "clip_count":     len(video_paths),
            "total_duration": f"{total_duration:.1f}s",
            "transition":     transition,
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}
