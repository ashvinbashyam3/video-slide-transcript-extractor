#!/usr/bin/env python3
"""
Create HTML Report from Transcript and Slides

Integrates a transcript with slide images to create a single HTML document.
Uses slide timestamps from metadata to insert slides at the correct position.
Uses Gemini Vision API to generate captions for each slide.
"""

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Load .env file if present
try:
    from dotenv import load_dotenv
    script_dir = Path(__file__).parent
    env_path = script_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai not installed. Run: pip install google-generativeai")
    sys.exit(1)


def parse_transcript(transcript_path: Path) -> List[Dict]:
    """
    Parse transcript file into list of entries with timestamps.

    Returns list of dicts: [{timestamp_seconds, timestamp_str, speaker, text, line_num}]
    """
    with open(transcript_path, 'r', encoding='utf-8') as f:
        content = f.read()

    entries = []
    lines = content.split('\n')

    # Pattern for timestamp and speaker: [MM:SS] SPEAKER: or [HH:MM:SS] SPEAKER:
    timestamp_pattern = r'^\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*([A-Z]+):\s*(.*)$'

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        match = re.match(timestamp_pattern, line)
        if match:
            ts_str = match.group(1)
            speaker = match.group(2)
            text = match.group(3)

            # Parse timestamp to seconds
            parts = ts_str.split(':')
            if len(parts) == 2:
                seconds = int(parts[0]) * 60 + int(parts[1])
            else:
                seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

            entries.append({
                'timestamp_seconds': seconds,
                'timestamp_str': ts_str,
                'speaker': speaker,
                'text': text,
                'line_num': i + 1
            })

    return entries


def load_slides_with_metadata(slides_dir: Path) -> List[Dict]:
    """
    Load slide images and their metadata (including timestamps).

    Returns list of dicts: [{path, filename, index, timestamp_sec, timestamp_str}]
    """
    slides = []

    # Try to load metadata JSON
    metadata_path = slides_dir / "slides_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        for slide_meta in metadata.get('slides', []):
            slide_path = slides_dir / slide_meta['filename']
            if slide_path.exists():
                slides.append({
                    'path': slide_path,
                    'filename': slide_meta['filename'],
                    'index': slide_meta['index'],
                    'timestamp_sec': slide_meta['timestamp_sec'],
                    'timestamp_str': slide_meta['timestamp_str']
                })
    else:
        # Fallback: load slides without timestamps
        print("Warning: No slides_metadata.json found. Slides will be distributed evenly.")
        for f in sorted(slides_dir.glob('slide_*.jpg')):
            match = re.search(r'slide_(\d+)', f.name)
            if match:
                index = int(match.group(1))
                slides.append({
                    'path': f,
                    'filename': f.name,
                    'index': index,
                    'timestamp_sec': None,
                    'timestamp_str': None
                })

    return slides


def image_to_base64(image_path: Path) -> str:
    """Convert image file to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_slide_captions(
    slides: List[Dict],
    api_key: str,
    verbose: bool = True
) -> List[Dict]:
    """
    Use Gemini Vision to generate captions for each slide.
    Returns slides with added 'caption' field.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    for i, slide in enumerate(slides):
        if verbose:
            print(f"  Generating caption for slide {i+1}/{len(slides)}: {slide['filename']}...")

        try:
            with open(slide['path'], 'rb') as f:
                image_data = f.read()

            prompt = """Analyze this presentation slide and provide a brief caption (2-3 sentences) describing what the slide shows. Focus on the main topic, key data points, or conclusions presented. Be concise and informative."""

            response = model.generate_content([
                {'mime_type': 'image/jpeg', 'data': image_data},
                prompt
            ])

            slide['caption'] = response.text.strip()

        except Exception as e:
            print(f"    Warning: Could not generate caption for {slide['filename']}: {e}")
            slide['caption'] = f"Slide {slide['index']}"

    return slides


def match_slides_to_transcript_by_timestamp(
    slides: List[Dict],
    entries: List[Dict],
    verbose: bool = True
) -> List[Dict]:
    """
    Match slides to transcript entries based on slide timestamps.
    Each slide is inserted after the transcript entry that comes just before
    the slide's timestamp (at paragraph boundaries).

    Returns slides with added 'insert_after_entry' field (index into entries).
    """
    if verbose:
        print("Matching slides to transcript by timestamp...")

    for slide in slides:
        slide_time = slide.get('timestamp_sec')

        if slide_time is None:
            # No timestamp - will be handled later
            slide['insert_after_entry'] = None
            continue

        # Find the last transcript entry that starts before this slide's timestamp
        best_entry_idx = 0
        for i, entry in enumerate(entries):
            if entry['timestamp_seconds'] <= slide_time:
                best_entry_idx = i
            else:
                break

        slide['insert_after_entry'] = best_entry_idx

        if verbose:
            print(f"  Slide {slide['index']} ({slide['timestamp_str']}) -> after entry at [{entries[best_entry_idx]['timestamp_str']}]")

    # Handle slides without timestamps (distribute evenly)
    no_timestamp_slides = [s for s in slides if s.get('insert_after_entry') is None]
    if no_timestamp_slides:
        interval = len(entries) // (len(no_timestamp_slides) + 1)
        for i, slide in enumerate(no_timestamp_slides):
            slide['insert_after_entry'] = min((i + 1) * interval, len(entries) - 1)

    return slides


def generate_html(
    entries: List[Dict],
    slides: List[Dict],
    output_path: Path,
    embed_images: bool = True,
    verbose: bool = True
) -> None:
    """
    Generate HTML document with transcript and slides.
    Slides are inserted in order after the appropriate transcript entry.
    """
    if verbose:
        print("Generating HTML document...")

    # Create a mapping of entry index -> slides to insert after
    slides_by_entry = {}
    for slide in slides:
        idx = slide.get('insert_after_entry', 0)
        if idx not in slides_by_entry:
            slides_by_entry[idx] = []
        slides_by_entry[idx].append(slide)

    # Sort slides within each entry by their original index (maintains presentation order)
    for idx in slides_by_entry:
        slides_by_entry[idx].sort(key=lambda s: s['index'])

    # HTML template
    html_parts = ['''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Webinar Transcript with Slides</title>
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .transcript-entry {
            background: white;
            padding: 15px 20px;
            margin: 10px 0;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .timestamp {
            color: #666;
            font-size: 0.85em;
            font-family: monospace;
            margin-right: 10px;
        }
        .speaker {
            font-weight: 600;
            color: #333;
            margin-right: 5px;
        }
        .speaker-JOSH { color: #2563eb; }
        .speaker-KARTIK { color: #059669; }
        .speaker-STEPHANIE { color: #7c3aed; }
        .speaker-SALVADOR { color: #dc2626; }
        .text {
            color: #333;
        }
        .slide-container {
            background: #1a1a2e;
            padding: 20px;
            margin: 20px 0;
            border-radius: 12px;
            text-align: center;
        }
        .slide-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .slide-header {
            color: #aaa;
            font-size: 0.8em;
            margin-bottom: 10px;
            font-family: monospace;
        }
        .slide-caption {
            color: #ccc;
            font-size: 0.9em;
            margin-top: 12px;
            line-height: 1.5;
            text-align: left;
            padding: 0 10px;
        }
        h1 {
            color: #1a1a2e;
            border-bottom: 3px solid #2563eb;
            padding-bottom: 10px;
        }
        .header-info {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .header-info h2 {
            margin-top: 0;
            color: #333;
        }
        .speaker-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 15px;
        }
        .speaker-legend span {
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 500;
        }
        .legend-JOSH { background: #dbeafe; color: #2563eb; }
        .legend-KARTIK { background: #d1fae5; color: #059669; }
        .legend-STEPHANIE { background: #ede9fe; color: #7c3aed; }
        .legend-SALVADOR { background: #fee2e2; color: #dc2626; }
    </style>
</head>
<body>
    <h1>Encoded Therapeutics Webinar</h1>
    <div class="header-info">
        <h2>ETX-101 for Dravet Syndrome - Clinical Data Presentation</h2>
        <div class="speaker-legend">
            <span class="legend-JOSH">Josh (Host)</span>
            <span class="legend-KARTIK">Kartik Ramamurthy (CEO)</span>
            <span class="legend-STEPHANIE">Stephanie Tagliatella (CSO)</span>
            <span class="legend-SALVADOR">Salvador Rico (CMO)</span>
        </div>
    </div>
''']

    # Generate transcript with slides interspersed
    for i, entry in enumerate(entries):
        # Add transcript entry
        speaker_class = f"speaker-{entry['speaker']}"
        html_parts.append(f'''
    <div class="transcript-entry">
        <span class="timestamp">[{entry['timestamp_str']}]</span>
        <span class="speaker {speaker_class}">{entry['speaker']}:</span>
        <span class="text">{entry['text']}</span>
    </div>
''')

        # Add any slides that come after this entry
        if i in slides_by_entry:
            for slide in slides_by_entry[i]:
                if embed_images:
                    img_b64 = image_to_base64(slide['path'])
                    img_src = f"data:image/jpeg;base64,{img_b64}"
                else:
                    img_src = slide['path'].name

                caption = slide.get('caption', f"Slide {slide['index']}")
                slide_time = slide.get('timestamp_str', '')
                time_display = f" at {slide_time}" if slide_time else ""

                html_parts.append(f'''
    <div class="slide-container">
        <div class="slide-header">Slide {slide['index']}{time_display}</div>
        <img src="{img_src}" alt="Slide {slide['index']}">
        <div class="slide-caption">{caption}</div>
    </div>
''')

    # Close HTML
    html_parts.append('''
</body>
</html>
''')

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(html_parts))

    if verbose:
        print(f"HTML report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create HTML report from transcript and slides"
    )

    parser.add_argument("transcript", help="Path to transcript text file")
    parser.add_argument("slides_dir", help="Path to directory containing slide images")
    parser.add_argument(
        "--output", "-o",
        help="Output HTML file path (default: transcript_with_slides.html)"
    )
    parser.add_argument(
        "--api-key", "-k",
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Don't embed images as base64 (reference external files instead)"
    )
    parser.add_argument(
        "--skip-captions",
        action="store_true",
        help="Skip Gemini caption generation (use generic captions)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    verbose = not args.quiet

    # Paths
    transcript_path = Path(args.transcript).resolve()
    slides_dir = Path(args.slides_dir).resolve()

    if not transcript_path.exists():
        print(f"Error: Transcript file not found: {transcript_path}")
        sys.exit(1)

    if not slides_dir.exists() or not slides_dir.is_dir():
        print(f"Error: Slides directory not found: {slides_dir}")
        sys.exit(1)

    # Output path
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = transcript_path.parent / f"{transcript_path.stem}_with_slides.html"

    # Parse transcript
    if verbose:
        print(f"Parsing transcript: {transcript_path}")
    entries = parse_transcript(transcript_path)
    if verbose:
        print(f"  Found {len(entries)} transcript entries")

    # Load slides with metadata (timestamps)
    if verbose:
        print(f"Loading slides from: {slides_dir}")
    slides = load_slides_with_metadata(slides_dir)
    if verbose:
        print(f"  Found {len(slides)} slides with timestamp metadata")

    if not slides:
        print("Warning: No slides found. Creating transcript-only HTML.")
        slides = []

    # Generate captions with Gemini (unless skipped)
    if slides and not args.skip_captions:
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("Warning: No Gemini API key provided. Using generic captions.")
            for slide in slides:
                slide['caption'] = f"Slide {slide['index']}"
        else:
            if verbose:
                print("Generating slide captions with Gemini Vision...")
            slides = generate_slide_captions(slides, api_key, verbose=verbose)
    else:
        for slide in slides:
            slide['caption'] = f"Slide {slide['index']}"

    # Match slides to transcript by timestamp
    if slides:
        slides = match_slides_to_transcript_by_timestamp(slides, entries, verbose=verbose)

    # Generate HTML
    generate_html(
        entries,
        slides,
        output_path,
        embed_images=not args.no_embed,
        verbose=verbose
    )

    if verbose:
        print(f"\nDone! Output: {output_path}")
        print(f"  - {len(entries)} transcript entries")
        print(f"  - {len(slides)} slides embedded (in presentation order)")


if __name__ == "__main__":
    main()
