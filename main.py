#!/usr/bin/env python3
"""
Video Slide & Transcript Extractor

A complete solution for extracting presentation slides and generating transcripts
from video recordings (e.g., Zoom webinars, presentations). Produces a single
self-contained HTML document combining slides and transcript.

Usage:
    python main.py                    # Opens file dialog
    python main.py path/to/video.mp4  # Process specific file

Output:
    Creates an HTML file with the same name as the video in the same directory.
    Example: video.mp4 -> video.html
"""

import argparse
import base64
import io
import os
import sys
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import tkinter as tk
from tkinter import filedialog

# Third-party imports
try:
    import cv2
    import numpy as np
except ImportError:
    print("Error: OpenCV not installed. Run: pip install opencv-python numpy")
    sys.exit(1)

try:
    from moviepy import VideoFileClip
except ImportError:
    print("Error: moviepy not installed. Run: pip install moviepy")
    sys.exit(1)

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai not installed. Run: pip install google-generativeai")
    sys.exit(1)

# Load .env file for API key
try:
    from dotenv import load_dotenv
    script_dir = Path(__file__).parent
    env_path = script_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


# =============================================================================
# SLIDE EXTRACTION
# =============================================================================

@dataclass
class SlideRegion:
    """Represents a detected slide region in the video frame."""
    x: int
    y: int
    width: int
    height: int

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Crop the frame to this region."""
        return frame[self.y:self.y + self.height, self.x:self.x + self.width]

    def area(self) -> int:
        return self.width * self.height


def compute_phash(image: np.ndarray, hash_size: int = 16) -> np.ndarray:
    """
    Compute perceptual hash of an image using DCT (Discrete Cosine Transform).

    The perceptual hash algorithm:
    1. Resize image to hash_size x hash_size (default 16x16)
    2. Convert to grayscale
    3. Apply DCT to get frequency components
    4. Take low-frequency 8x8 block (captures image structure, ignores fine details)
    5. Compute median and create binary hash (above/below median)

    This is robust to:
    - Minor compression artifacts
    - Small color/brightness changes
    - Slight geometric variations
    """
    resized = cv2.resize(image, (hash_size, hash_size), interpolation=cv2.INTER_AREA)

    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    dct = cv2.dct(np.float32(gray))
    dct_low = dct[:8, :8]
    median = np.median(dct_low)
    return (dct_low > median).flatten()


def phash_distance(hash1: np.ndarray, hash2: np.ndarray) -> float:
    """
    Compute Hamming distance between two perceptual hashes.
    Returns value between 0 (identical) and 1 (completely different).
    """
    return np.sum(hash1 != hash2) / len(hash1)


def is_slide_visible(frame: np.ndarray, region: SlideRegion) -> bool:
    """
    Determine if a slide is visible in the region (vs just a speaker view).

    Detection criteria:
    1. White/light background ratio - slides typically have white backgrounds
    2. Edge density - text and graphics create specific edge patterns
    3. Skin tone ratio - speaker views have more skin-colored pixels
    4. Local variance - slides have more uniform regions
    """
    cropped = region.crop(frame)
    if cropped.size == 0:
        return False

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Check for white/light background
    white_ratio = np.sum(gray > 200) / gray.size
    light_ratio = np.sum(gray > 180) / gray.size

    # Edge analysis for text detection
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Skin tone detection (HSV-based)
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    skin_mask = (
        (hsv[:, :, 0] >= 0) & (hsv[:, :, 0] <= 25) &
        (hsv[:, :, 1] >= 40) & (hsv[:, :, 1] <= 180) &
        (hsv[:, :, 2] >= 80)
    )
    skin_ratio = np.sum(skin_mask) / skin_mask.size

    # Decision: require light background, low skin, appropriate edge density
    has_white_background = white_ratio > 0.25
    has_light_background = light_ratio > 0.40
    has_slide_background = has_white_background or has_light_background
    has_low_skin = skin_ratio < 0.12
    has_content = edge_density > 0.015 and edge_density < 0.20

    return has_slide_background and has_low_skin and has_content


def find_slide_region(cap: cv2.VideoCapture, num_samples: int = 30) -> SlideRegion:
    """
    Analyze video frames to detect the slide region.

    Strategy:
    1. Sample frames throughout the video
    2. Find frames that look like they contain slides (white content)
    3. Analyze edge profiles to find slide boundaries
    4. Snap to standard aspect ratio (16:9, 16:10, or 4:3)
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ASPECT_RATIOS = {"16:9": 16/9, "16:10": 16/10, "4:3": 4/3}

    # Sample frames
    sample_positions = np.linspace(total_frames * 0.15, total_frames * 0.85, num_samples, dtype=int)

    slide_frames = []
    for pos in sample_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            white_ratio = np.sum(gray > 200) / gray.size
            if white_ratio > 0.2:
                slide_frames.append(frame)

    if not slide_frames:
        # Fallback: assume 16:9 slide taking 70% of width
        slide_width = int(width * 0.70)
        slide_height = int(slide_width / ASPECT_RATIOS["16:9"])
        margin_y = (height - slide_height) // 2
        return SlideRegion(0, margin_y, slide_width, slide_height)

    ref_frame = slide_frames[len(slide_frames) // 2]
    gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

    # Find right edge of slide (where speaker thumbnail begins)
    top_section = gray[:height // 3, :]
    white_in_top = np.sum(top_section > 200, axis=0)
    white_ratio_by_col = white_in_top / (height // 3)

    search_start = int(width * 0.60)
    search_end = int(width * 0.90)

    slide_right = int(width * 0.70)
    for x in range(search_end, search_start, -1):
        if white_ratio_by_col[x] > 0.3:
            slide_right = x + 10
            break

    # Find top/bottom boundaries
    row_mean = np.mean(gray[:, :slide_right], axis=1)

    top_boundary = 0
    for i in range(height // 6):
        if row_mean[i] > 150:
            top_boundary = max(0, i - 2)
            break

    bottom_boundary = height
    for i in range(height - 1, 5 * height // 6, -1):
        if row_mean[i] > 150:
            bottom_boundary = min(height, i + 2)
            break

    # Snap to standard aspect ratio
    detected_width = slide_right
    detected_height = bottom_boundary - top_boundary
    detected_ratio = detected_width / detected_height if detected_height > 0 else 1.5

    best_ratio_name = "16:9"
    best_diff = float('inf')
    for name, ratio in ASPECT_RATIOS.items():
        diff = abs(detected_ratio - ratio)
        if diff < best_diff:
            best_diff = diff
            best_ratio_name = name
    target_ratio = ASPECT_RATIOS[best_ratio_name]

    final_width = slide_right
    ideal_height = int(final_width / target_ratio)

    if ideal_height <= (bottom_boundary - top_boundary):
        final_height = ideal_height
        vertical_margin = ((bottom_boundary - top_boundary) - final_height) // 2
        final_top = top_boundary + vertical_margin
    else:
        final_height = bottom_boundary - top_boundary
        final_width = int(final_height * target_ratio)
        final_top = top_boundary

    return SlideRegion(0, final_top, min(final_width, width), min(final_height, height - final_top))


def extract_slides_to_memory(
    video_path: str,
    hash_threshold: float = 0.15,
    sample_interval_sec: float = 2.0,
    min_slide_duration_sec: float = 3.0,
    transition_delay_ms: int = 500,
    progress_callback=None
) -> List[Dict]:
    """
    Extract slides from video and return them in memory (not saved to disk).

    Returns list of dicts: [{image_data, timestamp_sec, timestamp_str, index}]

    Slide Change Detection Algorithm:
    1. Sample video at regular intervals (every 2 seconds by default)
    2. For each frame:
       a. Crop to detected slide region
       b. Check if it looks like a slide (vs speaker-only view)
       c. Compute perceptual hash
       d. Compare with previous frame's hash
       e. If hash distance > threshold, a slide change is detected
    3. After detecting change, wait for transition to complete (500ms)
    4. Verify stability by checking consecutive frames are similar
    5. Skip duplicate slides that appeared earlier
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    sample_interval = max(1, int(sample_interval_sec * fps))
    min_slide_frames = int(min_slide_duration_sec * fps)
    transition_delay_frames = int((transition_delay_ms / 1000.0) * fps)

    if progress_callback:
        progress_callback("Analyzing video to detect slide region...")

    slide_region = find_slide_region(cap, num_samples=30)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    slides = []
    slide_hashes = []
    last_hash = None
    last_slide_frame = -min_slide_frames
    frame_num = 0

    if progress_callback:
        progress_callback("Extracting slides...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % sample_interval != 0:
            frame_num += 1
            continue

        slide_crop = slide_region.crop(frame)
        if slide_crop.size == 0:
            frame_num += 1
            continue

        if not is_slide_visible(frame, slide_region):
            frame_num += 1
            continue

        current_hash = compute_phash(slide_crop)

        is_new = False
        if last_hash is None:
            is_new = True
        else:
            distance = phash_distance(current_hash, last_hash)
            if distance > hash_threshold:
                is_new = True

        if is_new and frame_num - last_slide_frame >= min_slide_frames:
            # Check not duplicate of earlier slide
            is_duplicate = any(phash_distance(current_hash, h) < hash_threshold for h in slide_hashes)

            if not is_duplicate:
                capture_frame_num = frame_num + transition_delay_frames

                if capture_frame_num < total_frames - 10:
                    current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, capture_frame_num)

                    # Find stable frame after transition
                    stable_crop = None
                    prev_hash = None

                    for _ in range(10):
                        ret_check, frame_check = cap.read()
                        if not ret_check:
                            break

                        crop_check = slide_region.crop(frame_check)
                        if crop_check.size == 0:
                            continue

                        current_check_hash = compute_phash(crop_check)

                        if prev_hash is not None:
                            stability_dist = phash_distance(current_check_hash, prev_hash)
                            if stability_dist < 0.05:
                                stable_crop = crop_check
                                capture_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                                break

                        prev_hash = current_check_hash
                        for _ in range(2):
                            cap.read()

                    if stable_crop is None and prev_hash is not None:
                        stable_crop = crop_check

                    if stable_crop is not None and stable_crop.size > 0:
                        if is_slide_visible(frame_check, slide_region):
                            # Encode image to JPEG in memory
                            _, img_encoded = cv2.imencode('.jpg', stable_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            img_bytes = img_encoded.tobytes()

                            timestamp_sec = capture_frame_num / fps if fps > 0 else 0
                            minutes = int(timestamp_sec // 60)
                            seconds = int(timestamp_sec % 60)

                            slides.append({
                                'image_data': img_bytes,
                                'timestamp_sec': timestamp_sec,
                                'timestamp_str': f"{minutes:02d}:{seconds:02d}",
                                'index': len(slides) + 1
                            })

                            slide_hashes.append(compute_phash(stable_crop))
                            last_slide_frame = frame_num

                            if progress_callback:
                                progress_callback(f"  Found slide {len(slides)} at {minutes:02d}:{seconds:02d}")

                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_pos))

        last_hash = current_hash
        frame_num += 1

        # Progress update
        if progress_callback and frame_num % (sample_interval * 50) == 0:
            progress = frame_num / total_frames * 100
            progress_callback(f"  Progress: {progress:.0f}%")

    cap.release()
    return slides


# =============================================================================
# TRANSCRIPT GENERATION
# =============================================================================

def extract_audio_chunk(video_path: str, start_sec: float, end_sec: float) -> bytes:
    """Extract audio chunk from video and return as MP3 bytes."""
    video = VideoFileClip(video_path)
    audio_clip = video.audio.subclipped(start_sec, min(end_sec, video.duration))

    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
        tmp_path = tmp.name

    audio_clip.write_audiofile(tmp_path, codec='mp3', bitrate='128k')
    audio_clip.close()
    video.close()

    # Read back and delete
    with open(tmp_path, 'rb') as f:
        audio_bytes = f.read()
    os.unlink(tmp_path)

    return audio_bytes


def generate_transcript_gemini(
    video_path: str,
    api_key: str,
    chunk_duration_min: int = 20,
    progress_callback=None
) -> List[Dict]:
    """
    Generate transcript using Gemini API with speaker diarization.

    Returns list of dicts: [{speaker, text}] (no timestamps for final output)

    Process:
    1. Split audio into chunks (20 min each for long videos)
    2. Upload each chunk to Gemini File API
    3. Generate transcript with speaker identification
    4. Combine chunks and clean up speaker names
    """
    genai.configure(api_key=api_key)

    video = VideoFileClip(video_path)
    duration = video.duration
    video.close()

    chunk_duration_sec = chunk_duration_min * 60
    num_chunks = max(1, int(np.ceil(duration / chunk_duration_sec)))

    if progress_callback:
        progress_callback(f"Generating transcript ({num_chunks} audio chunks)...")

    all_entries = []

    for chunk_idx in range(num_chunks):
        start_sec = chunk_idx * chunk_duration_sec
        end_sec = min((chunk_idx + 1) * chunk_duration_sec, duration)

        if progress_callback:
            progress_callback(f"  Processing audio chunk {chunk_idx + 1}/{num_chunks}...")

        # Extract audio chunk
        audio_bytes = extract_audio_chunk(video_path, start_sec, end_sec)

        # Upload to Gemini
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        uploaded_file = genai.upload_file(tmp_path)
        os.unlink(tmp_path)

        # Wait for processing
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = genai.get_file(uploaded_file.name)

        if uploaded_file.state.name != "ACTIVE":
            continue

        # Generate transcript
        model = genai.GenerativeModel('gemini-2.0-flash')

        prompt = """Transcribe this audio with speaker diarization.

Instructions:
- Identify different speakers and label them as SPEAKER_1, SPEAKER_2, etc.
- Format each speaker turn as: SPEAKER_X: [their speech]
- Include all spoken content
- One speaker turn per line

Output the transcript:"""

        response = model.generate_content(
            [uploaded_file, prompt],
            generation_config=genai.GenerationConfig(temperature=0.1)
        )

        # Parse transcript
        if response.text:
            for line in response.text.strip().split('\n'):
                line = line.strip()
                if ':' in line and line.split(':')[0].strip().startswith('SPEAKER'):
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        speaker = parts[0].strip()
                        text = parts[1].strip()
                        if text:
                            all_entries.append({
                                'speaker': speaker,
                                'text': text,
                                'chunk_idx': chunk_idx
                            })

        # Cleanup
        try:
            genai.delete_file(uploaded_file.name)
        except:
            pass

    return all_entries


def cleanup_transcript_entries(
    entries: List[Dict],
    api_key: str,
    progress_callback=None
) -> List[Dict]:
    """
    Clean up transcript using Gemini to identify speakers and fix errors.
    """
    if not entries:
        return entries

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    # Process in chunks of 15 entries
    chunk_size = 15
    cleaned_entries = []

    if progress_callback:
        progress_callback("Cleaning up transcript...")

    for i in range(0, len(entries), chunk_size):
        chunk = entries[i:i + chunk_size]
        chunk_text = '\n'.join([f"{e['speaker']}: {e['text']}" for e in chunk])

        prompt = f"""Clean up this transcript segment from a webinar/presentation.

Instructions:
1. Identify speakers by name if mentioned (replace SPEAKER_1, etc. with actual names)
2. Fix transcription errors and technical terms
3. Keep the format: SPEAKER_NAME: [text]
4. Do not add or remove content

Transcript:
{chunk_text}

Return cleaned transcript:"""

        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(temperature=0.1, max_output_tokens=8000)
            )

            if response.text:
                for line in response.text.strip().split('\n'):
                    line = line.strip()
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            speaker = parts[0].strip()
                            text = parts[1].strip()
                            if text and not speaker.startswith('['):
                                cleaned_entries.append({'speaker': speaker, 'text': text})
        except:
            # On error, keep original entries
            cleaned_entries.extend(chunk)

    return cleaned_entries if cleaned_entries else entries


# =============================================================================
# SLIDE CAPTIONS
# =============================================================================

def generate_slide_captions(
    slides: List[Dict],
    api_key: str,
    progress_callback=None
) -> List[Dict]:
    """Generate captions for slides using Gemini Vision."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    if progress_callback:
        progress_callback("Generating slide captions...")

    for i, slide in enumerate(slides):
        if progress_callback:
            progress_callback(f"  Captioning slide {i + 1}/{len(slides)}...")

        try:
            prompt = "Describe this presentation slide in 2-3 sentences. Focus on the main topic and key data points."

            response = model.generate_content([
                {'mime_type': 'image/jpeg', 'data': slide['image_data']},
                prompt
            ])

            slide['caption'] = response.text.strip()
        except Exception as e:
            slide['caption'] = f"Slide {slide['index']}"

    return slides


# =============================================================================
# HTML GENERATION
# =============================================================================

def generate_html(
    transcript_entries: List[Dict],
    slides: List[Dict],
    title: str = "Video Transcript with Slides"
) -> str:
    """
    Generate self-contained HTML document with transcript and slides.

    Slides are inserted based on their timestamps, appearing after the
    transcript entry that precedes their appearance time.
    """

    # Map slides to insertion points based on rough timing
    # Since transcript entries don't have timestamps, distribute slides evenly
    # based on their relative position in the video

    if slides and transcript_entries:
        # Get max slide time
        max_slide_time = max(s['timestamp_sec'] for s in slides)
        entries_per_sec = len(transcript_entries) / max_slide_time if max_slide_time > 0 else 1

        for slide in slides:
            # Estimate which entry this slide should follow
            estimated_entry = int(slide['timestamp_sec'] * entries_per_sec)
            slide['insert_after_entry'] = min(estimated_entry, len(transcript_entries) - 1)

    # Create mapping of entry index -> slides
    slides_by_entry = {}
    for slide in slides:
        idx = slide.get('insert_after_entry', 0)
        if idx not in slides_by_entry:
            slides_by_entry[idx] = []
        slides_by_entry[idx].append(slide)

    # Sort slides at each entry by index
    for idx in slides_by_entry:
        slides_by_entry[idx].sort(key=lambda s: s['index'])

    # Generate HTML
    html_parts = [f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #1a1a2e;
            border-bottom: 3px solid #2563eb;
            padding-bottom: 10px;
        }}
        .transcript-entry {{
            background: white;
            padding: 15px 20px;
            margin: 10px 0;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .speaker {{
            font-weight: 600;
            color: #2563eb;
            margin-right: 8px;
        }}
        .text {{ color: #333; }}
        .slide-container {{
            background: #1a1a2e;
            padding: 20px;
            margin: 20px 0;
            border-radius: 12px;
            text-align: center;
        }}
        .slide-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        .slide-header {{
            color: #aaa;
            font-size: 0.8em;
            margin-bottom: 10px;
            font-family: monospace;
        }}
        .slide-caption {{
            color: #ccc;
            font-size: 0.9em;
            margin-top: 12px;
            line-height: 1.5;
            text-align: left;
            padding: 0 10px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
''']

    # Generate content
    for i, entry in enumerate(transcript_entries):
        speaker = entry.get('speaker', 'Speaker')
        text = entry.get('text', '')

        html_parts.append(f'''
    <div class="transcript-entry">
        <span class="speaker">{speaker}:</span>
        <span class="text">{text}</span>
    </div>
''')

        # Insert slides after this entry
        if i in slides_by_entry:
            for slide in slides_by_entry[i]:
                img_b64 = base64.b64encode(slide['image_data']).decode('utf-8')
                caption = slide.get('caption', f"Slide {slide['index']}")

                html_parts.append(f'''
    <div class="slide-container">
        <div class="slide-header">Slide {slide['index']}</div>
        <img src="data:image/jpeg;base64,{img_b64}" alt="Slide {slide['index']}">
        <div class="slide-caption">{caption}</div>
    </div>
''')

    html_parts.append('''
</body>
</html>
''')

    return ''.join(html_parts)


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def select_video_file() -> Optional[str]:
    """Open file dialog to select a video file."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Get Downloads folder path
    downloads_path = str(Path.home() / "Downloads")

    file_path = filedialog.askopenfilename(
        title="Select Video File",
        initialdir=downloads_path,
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("MP4 files", "*.mp4"),
            ("All files", "*.*")
        ]
    )

    root.destroy()
    return file_path if file_path else None


def print_progress(message: str):
    """Print progress message to console."""
    print(message)


def main():
    parser = argparse.ArgumentParser(
        description="Extract slides and transcript from video, generate HTML report"
    )
    parser.add_argument(
        "video",
        nargs="?",
        help="Path to video file (opens file dialog if not provided)"
    )
    parser.add_argument(
        "--api-key", "-k",
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Gemini API key required.")
        print("Set GEMINI_API_KEY environment variable or use --api-key")
        sys.exit(1)

    # Get video file
    if args.video:
        video_path = Path(args.video).resolve()
    else:
        print("Select a video file...")
        selected = select_video_file()
        if not selected:
            print("No file selected. Exiting.")
            sys.exit(0)
        video_path = Path(selected).resolve()

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    print(f"\nProcessing: {video_path.name}")
    print("=" * 60)

    start_time = time.time()

    # Step 1: Extract slides
    print("\n[1/4] Extracting slides from video...")
    slides = extract_slides_to_memory(str(video_path), progress_callback=print_progress)
    print(f"  Found {len(slides)} slides")

    # Step 2: Generate transcript
    print("\n[2/4] Generating transcript...")
    transcript = generate_transcript_gemini(str(video_path), api_key, progress_callback=print_progress)
    print(f"  Generated {len(transcript)} transcript entries")

    # Step 3: Clean up transcript
    print("\n[3/4] Cleaning up transcript...")
    transcript = cleanup_transcript_entries(transcript, api_key, progress_callback=print_progress)

    # Step 4: Generate slide captions
    print("\n[4/4] Generating slide captions...")
    slides = generate_slide_captions(slides, api_key, progress_callback=print_progress)

    # Generate HTML
    print("\nGenerating HTML document...")
    title = video_path.stem.replace('_', ' ').replace('-', ' ')
    html_content = generate_html(transcript, slides, title=title)

    # Save output
    output_path = video_path.parent / f"{video_path.stem}.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"Done! Output saved to: {output_path}")
    print(f"Processing time: {elapsed / 60:.1f} minutes")
    print(f"  - {len(slides)} slides extracted")
    print(f"  - {len(transcript)} transcript entries")


if __name__ == "__main__":
    main()
