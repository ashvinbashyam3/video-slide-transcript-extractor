#!/usr/bin/env python3
"""
Video Slide Extractor

Extracts presentation slides from video recordings (e.g., Zoom recordings).
Detects slide changes and saves each unique slide as a JPEG image.
Automatically crops to the slide region and ignores speaker-only frames.

Usage:
    python extract_slides.py <video_file>
    python extract_slides.py /path/to/video.mp4

Output:
    Creates a folder with the same name as the video file (without extension)
    in the same directory as the video, containing numbered JPEG images.
"""

import argparse
import cv2
import json
import numpy as np
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List


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
    Compute perceptual hash of an image.
    More robust to compression artifacts than pixel comparison.
    """
    # Resize to hash_size x hash_size
    resized = cv2.resize(image, (hash_size, hash_size), interpolation=cv2.INTER_AREA)

    # Convert to grayscale if needed
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    # Compute DCT
    dct = cv2.dct(np.float32(gray))

    # Use top-left 8x8 of DCT (low frequencies)
    dct_low = dct[:8, :8]

    # Compute median and create binary hash
    median = np.median(dct_low)
    return (dct_low > median).flatten()


def phash_distance(hash1: np.ndarray, hash2: np.ndarray) -> float:
    """
    Compute hamming distance between two perceptual hashes.
    Returns value between 0 (identical) and 1 (completely different).
    """
    return np.sum(hash1 != hash2) / len(hash1)


def compute_histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute histogram similarity between two images.
    Returns value between 0 and 1 (1 = identical histograms).
    """
    # Convert to HSV for better color comparison
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    # Compute histograms
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])

    # Normalize
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    # Compare using correlation
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def detect_slide_region_from_layout(frame: np.ndarray) -> Optional[SlideRegion]:
    """
    Detect slide region by analyzing the frame layout.
    Looks for the main content area, excluding speaker thumbnails.
    """
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # For typical Zoom/Teams recordings with speaker in corner:
    # The slide usually takes up the left ~70-85% of the frame

    # Method 1: Look for vertical edge that separates slide from speaker
    # Compute vertical edge strength
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.abs(sobel_x)

    # Sum along vertical axis to find strong vertical lines
    vertical_profile = np.sum(sobel_x, axis=0)

    # Look for a strong vertical edge in the right portion (60-90% of width)
    search_start = int(width * 0.60)
    search_end = int(width * 0.90)

    search_region = vertical_profile[search_start:search_end]
    if len(search_region) > 0:
        # Find peaks in the vertical edge profile
        threshold = np.mean(search_region) + 2 * np.std(search_region)
        peaks = np.where(search_region > threshold)[0]

        if len(peaks) > 0:
            # Use the leftmost strong vertical edge as the boundary
            slide_right = search_start + peaks[0]

            # Detect top/bottom boundaries (look for horizontal edges)
            sobel_y = cv2.Sobel(gray[:, :slide_right], cv2.CV_64F, 0, 1, ksize=3)
            sobel_y = np.abs(sobel_y)
            horizontal_profile = np.sum(sobel_y, axis=1)

            # Find top boundary
            top_threshold = np.mean(horizontal_profile[:height//4]) + 2 * np.std(horizontal_profile[:height//4])
            top_peaks = np.where(horizontal_profile[:height//4] > top_threshold)[0]
            slide_top = top_peaks[-1] if len(top_peaks) > 0 else 0

            # Find bottom boundary
            bottom_threshold = np.mean(horizontal_profile[3*height//4:]) + 2 * np.std(horizontal_profile[3*height//4:])
            bottom_peaks = np.where(horizontal_profile[3*height//4:] > bottom_threshold)[0]
            slide_bottom = (3*height//4 + bottom_peaks[0]) if len(bottom_peaks) > 0 else height

            return SlideRegion(0, slide_top, slide_right, slide_bottom - slide_top)

    # Fallback: assume standard layout (left 70% is slide, exclude top/bottom 5%)
    margin_y = int(height * 0.03)
    return SlideRegion(0, margin_y, int(width * 0.70), height - 2 * margin_y)


def is_slide_visible(frame: np.ndarray, region: SlideRegion) -> bool:
    """
    Determine if a slide is visible in the region (vs just a speaker view).
    Slides typically have:
    - White or light background with significant coverage
    - Text elements (horizontal edge patterns)
    - Geometric shapes and diagrams

    Speaker-only views have:
    - Varied, natural colors (skin tones)
    - No dominant white background
    - Different edge patterns (curves vs straight lines)
    """
    cropped = region.crop(frame)

    if cropped.size == 0:
        return False

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Check 1: White/light background - most presentation slides have white backgrounds
    white_ratio = np.sum(gray > 200) / gray.size
    light_ratio = np.sum(gray > 180) / gray.size

    # Check 2: Edge analysis for text detection
    # Text creates horizontal edge patterns
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Detect horizontal edges (text lines)
    sobel_h = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_v = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    h_edges = np.abs(sobel_h)
    v_edges = np.abs(sobel_v)

    # Text has more horizontal edges, faces have more varied edges
    h_edge_sum = np.sum(h_edges)
    v_edge_sum = np.sum(v_edges)
    edge_ratio = h_edge_sum / (v_edge_sum + 1e-6)

    # Check 3: Color analysis - slides have less skin tone colors
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    # Skin tone detection (rough heuristic)
    skin_mask = (
        (hsv[:, :, 0] >= 0) & (hsv[:, :, 0] <= 25) &  # Hue in skin range
        (hsv[:, :, 1] >= 40) & (hsv[:, :, 1] <= 180) &  # Some saturation
        (hsv[:, :, 2] >= 80)  # Not too dark
    )
    skin_ratio = np.sum(skin_mask) / skin_mask.size

    # Check 4: Look for rectangular regions with uniform color (slide backgrounds)
    # Slides often have large uniform regions
    blur = cv2.blur(gray.astype(np.float32), (31, 31))
    local_var = cv2.blur((gray.astype(np.float32) - blur) ** 2, (31, 31))
    local_var = np.maximum(local_var, 0)
    local_std = np.sqrt(local_var)

    # Percentage of image with very low variance (uniform regions)
    uniform_ratio = np.sum(local_std < 10) / local_std.size

    # Decision logic:
    # A slide should have:
    # - Significant white/light background (most slides have white backgrounds)
    # - Low skin tone presence
    # - Some edge content (text/graphics)

    # Require actual white/light background - not just any uniform color
    # This filters out speaker frames with uniform colored backgrounds (like blue)
    has_white_background = white_ratio > 0.25
    has_light_background = light_ratio > 0.40

    has_slide_background = has_white_background or has_light_background
    has_low_skin = skin_ratio < 0.12  # Stricter skin threshold
    has_content = edge_density > 0.015 and edge_density < 0.20

    is_slide = has_slide_background and has_low_skin and has_content

    return is_slide


def find_slide_region_from_samples(cap: cv2.VideoCapture, num_samples: int = 30,
                                   target_aspect_ratio: str = "auto") -> SlideRegion:
    """
    Analyze multiple frames to determine the slide region.
    Uses standard PowerPoint aspect ratios for proper framing.

    Args:
        cap: Video capture object
        num_samples: Number of frames to sample
        target_aspect_ratio: "16:9", "16:10", "4:3", or "auto" to detect
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Standard PowerPoint aspect ratios
    ASPECT_RATIOS = {
        "16:9": 16 / 9,    # 1.778 - Modern widescreen
        "16:10": 16 / 10,  # 1.600 - Common widescreen
        "4:3": 4 / 3,      # 1.333 - Traditional/legacy
    }

    # Sample frames throughout the video
    sample_positions = np.linspace(total_frames * 0.15, total_frames * 0.85, num_samples, dtype=int)

    # Collect frames that look like they have slides
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

    # Use a representative frame for analysis
    ref_frame = slide_frames[len(slide_frames) // 2]
    gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

    # Step 1: Find the right edge of the slide area
    # Look for where the speaker thumbnail begins (usually top-right)
    # The slide area typically has more uniform/white content

    # Analyze the top portion of the frame (where speaker thumbnail usually is)
    top_section = gray[:height // 3, :]

    # For each column, calculate the variance in the top section
    col_variance_top = np.var(top_section, axis=0)

    # Smooth the variance profile
    kernel_size = 30
    col_variance_smooth = np.convolve(col_variance_top, np.ones(kernel_size)/kernel_size, mode='same')

    # Find where variance increases significantly (slide->speaker boundary)
    # Start from 60% of width and look for changes
    search_start = int(width * 0.60)
    search_end = int(width * 0.90)

    # Look for the point where non-white content appears (speaker thumbnail)
    # Count white pixels in each column of the top section
    white_in_top = np.sum(top_section > 200, axis=0)
    white_ratio_by_col = white_in_top / (height // 3)

    # Find where white ratio drops significantly (entering speaker area)
    slide_right = int(width * 0.70)  # Default
    for x in range(search_end, search_start, -1):
        if white_ratio_by_col[x] > 0.3:  # Found slide content
            slide_right = x + 10  # Add small margin
            break

    # Step 2: Find top and bottom boundaries
    # Look for header/footer bars (usually darker strips)
    row_mean = np.mean(gray[:, :slide_right], axis=1)

    # Find top boundary (after header bar)
    top_boundary = 0
    for i in range(height // 6):
        if row_mean[i] > 150:  # Found light content
            top_boundary = max(0, i - 2)
            break

    # Find bottom boundary (before footer bar)
    bottom_boundary = height
    for i in range(height - 1, 5 * height // 6, -1):
        if row_mean[i] > 150:  # Found light content
            bottom_boundary = min(height, i + 2)
            break

    # Step 3: Calculate detected dimensions
    detected_width = slide_right
    detected_height = bottom_boundary - top_boundary
    detected_ratio = detected_width / detected_height if detected_height > 0 else 1.5

    # Step 4: Snap to nearest standard aspect ratio
    if target_aspect_ratio == "auto":
        # Find closest standard ratio
        best_ratio_name = "16:9"
        best_diff = float('inf')
        for name, ratio in ASPECT_RATIOS.items():
            diff = abs(detected_ratio - ratio)
            if diff < best_diff:
                best_diff = diff
                best_ratio_name = name
        target_ratio = ASPECT_RATIOS[best_ratio_name]
    else:
        target_ratio = ASPECT_RATIOS.get(target_aspect_ratio, 16/9)
        best_ratio_name = target_aspect_ratio

    # Step 5: Adjust dimensions to match standard aspect ratio
    # Prioritize keeping the full width and adjusting height
    final_width = slide_right
    ideal_height = int(final_width / target_ratio)

    if ideal_height <= (bottom_boundary - top_boundary):
        # Height fits, center vertically
        final_height = ideal_height
        vertical_margin = ((bottom_boundary - top_boundary) - final_height) // 2
        final_top = top_boundary + vertical_margin
    else:
        # Need to reduce width to fit height
        final_height = bottom_boundary - top_boundary
        final_width = int(final_height * target_ratio)
        final_top = top_boundary

    # Ensure we don't exceed frame boundaries
    final_width = min(final_width, width)
    final_height = min(final_height, height - final_top)

    return SlideRegion(0, final_top, final_width, final_height)


def is_duplicate_slide(new_hash: np.ndarray, existing_hashes: List[np.ndarray],
                       threshold: float = 0.15) -> bool:
    """
    Check if a slide is a duplicate of any existing slide.
    """
    for existing_hash in existing_hashes:
        distance = phash_distance(new_hash, existing_hash)
        if distance < threshold:
            return True
    return False


def extract_slides(video_path: str, output_dir: str,
                   hash_threshold: float = 0.15,
                   sample_interval_sec: float = 2.0,
                   min_slide_duration_sec: float = 3.0,
                   transition_delay_ms: int = 500,
                   aspect_ratio: str = "auto",
                   verbose: bool = True) -> int:
    """
    Extract slides from a video file.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save extracted slides
        hash_threshold: Perceptual hash distance threshold (0-1, lower = stricter)
        sample_interval_sec: Check every N seconds for changes
        min_slide_duration_sec: Minimum seconds a slide must appear to be saved
        transition_delay_ms: Delay in milliseconds after detecting slide change before capture
        aspect_ratio: Target aspect ratio ("16:9", "16:10", "4:3", or "auto")
        verbose: Print progress information

    Returns:
        Number of slides extracted
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sample_interval = max(1, int(sample_interval_sec * fps))
    min_slide_frames = int(min_slide_duration_sec * fps)
    transition_delay_frames = int((transition_delay_ms / 1000.0) * fps)

    # Start timing
    start_time = time.time()

    if verbose:
        duration_sec = total_frames / fps if fps > 0 else 0
        print(f"Video: {width}x{height}, {total_frames} frames, {fps:.1f} fps")
        print(f"Duration: {duration_sec:.1f}s ({duration_sec/60:.1f} minutes)")
        print(f"Sampling every {sample_interval_sec}s ({sample_interval} frames)")
        print(f"Transition delay: {transition_delay_ms}ms ({transition_delay_frames} frames)")

    # Detect slide region
    if verbose:
        print("Analyzing video to detect slide region...")

    slide_region = find_slide_region_from_samples(cap, num_samples=30, target_aspect_ratio=aspect_ratio)

    # Calculate actual aspect ratio
    actual_ratio = slide_region.width / slide_region.height if slide_region.height > 0 else 0
    ratio_name = "unknown"
    if abs(actual_ratio - 16/9) < 0.1:
        ratio_name = "16:9"
    elif abs(actual_ratio - 16/10) < 0.1:
        ratio_name = "16:10"
    elif abs(actual_ratio - 4/3) < 0.1:
        ratio_name = "4:3"

    if verbose:
        print(f"Detected slide region: x={slide_region.x}, y={slide_region.y}, "
              f"w={slide_region.width}, h={slide_region.height}")
        print(f"Aspect ratio: {actual_ratio:.3f} ({ratio_name})")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    slides = []  # List of (frame_number, slide_image, hash)
    slide_hashes = []  # For duplicate detection
    last_hash = None
    last_slide_frame = -min_slide_frames
    frame_num = 0
    frames_processed = 0

    if verbose:
        print("Extracting slides...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only check at sample intervals
        if frame_num % sample_interval != 0:
            frame_num += 1
            continue

        frames_processed += 1

        # Crop to slide region
        slide_crop = slide_region.crop(frame)

        if slide_crop.size == 0:
            frame_num += 1
            continue

        # Check if this looks like a slide (not just speaker)
        if not is_slide_visible(frame, slide_region):
            frame_num += 1
            continue

        # Compute perceptual hash
        current_hash = compute_phash(slide_crop)

        # Check if this is a new slide
        is_new = False

        if last_hash is None:
            is_new = True
        else:
            # Compare with previous frame's hash
            distance = phash_distance(current_hash, last_hash)
            if distance > hash_threshold:
                is_new = True

        if is_new:
            # Check minimum duration since last slide
            if frame_num - last_slide_frame >= min_slide_frames:
                # Check if it's not a duplicate of any previous slide
                if not is_duplicate_slide(current_hash, slide_hashes, hash_threshold):
                    # Apply transition delay - seek ahead to capture after transition completes
                    capture_frame_num = frame_num + transition_delay_frames

                    if capture_frame_num < total_frames - 10:
                        # Save current position
                        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

                        # Seek to delayed position
                        cap.set(cv2.CAP_PROP_POS_FRAMES, capture_frame_num)

                        # Stability detection: find a stable frame after the transition
                        # Read consecutive frames and wait until they're similar (transition complete)
                        stable_frame = None
                        stable_crop = None
                        prev_hash = None
                        stability_checks = 10  # Check up to 10 frames for stability
                        frames_to_skip = 3  # Check every 3rd frame (~120ms at 25fps)

                        for stability_i in range(stability_checks):
                            ret_check, frame_check = cap.read()
                            if not ret_check:
                                break

                            crop_check = slide_region.crop(frame_check)
                            if crop_check.size == 0:
                                continue

                            current_check_hash = compute_phash(crop_check)

                            if prev_hash is not None:
                                # Check if frame is stable (very similar to previous)
                                stability_dist = phash_distance(current_check_hash, prev_hash)
                                if stability_dist < 0.05:  # Very similar = stable
                                    stable_frame = frame_check
                                    stable_crop = crop_check
                                    capture_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                                    break

                            prev_hash = current_check_hash

                            # Skip a few frames for next check
                            for _ in range(frames_to_skip - 1):
                                cap.read()

                        # Use the stable frame if found, otherwise use last checked frame
                        if stable_crop is None and prev_hash is not None:
                            # No stable frame found, use the last one we checked
                            stable_crop = crop_check
                            stable_frame = frame_check

                        if stable_crop is not None and stable_crop.size > 0:
                            # Verify it's still a slide
                            if is_slide_visible(stable_frame, slide_region):
                                slide_num = len(slides) + 1
                                slides.append((capture_frame_num, stable_crop.copy()))
                                slide_hashes.append(compute_phash(stable_crop))
                                last_slide_frame = frame_num

                                if verbose:
                                    time_sec = capture_frame_num / fps if fps > 0 else 0
                                    print(f"  Slide {slide_num} at {time_sec:.1f}s (frame {capture_frame_num})")

                        # Restore position to continue scanning
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_pos))

        last_hash = current_hash
        frame_num += 1

        # Progress update
        if verbose and frames_processed % 100 == 0:
            progress = frame_num / total_frames * 100
            print(f"  Progress: {progress:.1f}%")

    cap.release()

    # Save slides and metadata
    if verbose:
        print(f"\nSaving {len(slides)} slides...")

    metadata = {
        "video_file": os.path.basename(video_path),
        "video_duration_sec": total_frames / fps if fps > 0 else 0,
        "fps": fps,
        "slide_region": {
            "x": slide_region.x,
            "y": slide_region.y,
            "width": slide_region.width,
            "height": slide_region.height
        },
        "slides": []
    }

    for i, (frame_num, slide) in enumerate(slides):
        slide_filename = f"slide_{i + 1:03d}.jpg"
        output_path = os.path.join(output_dir, slide_filename)
        cv2.imwrite(output_path, slide, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Calculate timestamp
        timestamp_sec = frame_num / fps if fps > 0 else 0
        minutes = int(timestamp_sec // 60)
        seconds = int(timestamp_sec % 60)
        timestamp_str = f"{minutes:02d}:{seconds:02d}"

        slide_meta = {
            "index": i + 1,
            "filename": slide_filename,
            "frame_num": frame_num,
            "timestamp_sec": round(timestamp_sec, 2),
            "timestamp_str": timestamp_str
        }
        metadata["slides"].append(slide_meta)

        if verbose:
            print(f"  Saved: {output_path} (at {timestamp_str})")

    # Save metadata JSON
    metadata_path = os.path.join(output_dir, "slides_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"  Saved metadata: {metadata_path}")

    # Calculate runtime
    end_time = time.time()
    runtime_sec = end_time - start_time

    if verbose:
        print(f"\nDone! Extracted {len(slides)} slides to {output_dir}")
        print(f"Runtime: {runtime_sec:.1f}s ({runtime_sec/60:.2f} minutes)")

    return len(slides)


def main():
    parser = argparse.ArgumentParser(
        description="Extract presentation slides from video recordings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_slides.py video.mp4
  python extract_slides.py /path/to/recording.mp4 --threshold 0.20
  python extract_slides.py meeting.mp4 --interval 3 --quiet
        """
    )

    parser.add_argument("video", help="Path to the video file")
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.15,
        help="Hash distance threshold for detecting changes (0-1, default: 0.15)"
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=2.0,
        help="Sampling interval in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--min-duration", "-d",
        type=float,
        default=3.0,
        help="Minimum slide duration in seconds (default: 3.0)"
    )
    parser.add_argument(
        "--transition-delay", "-T",
        type=int,
        default=500,
        help="Delay in ms after slide change before capture, to skip transitions (default: 500)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory (default: same directory as video, named after video)"
    )
    parser.add_argument(
        "--aspect-ratio", "-a",
        choices=["auto", "16:9", "16:10", "4:3"],
        default="auto",
        help="Slide aspect ratio: auto (detect), 16:9, 16:10, or 4:3 (default: auto)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Validate input
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        output_dir = video_path.parent / video_path.stem

    try:
        count = extract_slides(
            str(video_path),
            str(output_dir),
            hash_threshold=args.threshold,
            sample_interval_sec=args.interval,
            min_slide_duration_sec=args.min_duration,
            transition_delay_ms=args.transition_delay,
            aspect_ratio=args.aspect_ratio,
            verbose=not args.quiet
        )

        if count == 0:
            print("Warning: No slides were detected in the video.", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
