#!/usr/bin/env python3
"""
Transcript Cleanup Script using Gemini API

Cleans up a raw transcript by:
- Removing chunk labels/separators
- Identifying and naming speakers based on context
- Fixing typos, repeated words, and technical terms
- Merging timestamps from multiple chunks into continuous timeline
"""

import argparse
import os
import sys
import re
from pathlib import Path

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


def parse_chunk_info(transcript: str) -> list:
    """Extract chunk timing information from the transcript."""
    # Pattern: --- Chunk N (START - END) ---
    chunk_pattern = r'--- Chunk (\d+) \(([^)]+)\) ---'
    chunks = []

    for match in re.finditer(chunk_pattern, transcript):
        chunk_num = int(match.group(1))
        time_range = match.group(2)
        # Parse time range like "00:00 - 20:00" or "40:00 - 01:00:00"
        parts = time_range.split(' - ')
        if len(parts) == 2:
            start_str, end_str = parts
            chunks.append({
                'chunk': chunk_num,
                'start': parse_time(start_str),
                'end': parse_time(end_str),
                'start_str': start_str,
                'end_str': end_str
            })

    return chunks


def parse_time(time_str: str) -> int:
    """Parse time string to seconds."""
    parts = time_str.strip().split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0


def format_time(seconds: int) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def adjust_timestamps(transcript: str, chunks: list) -> str:
    """Adjust timestamps in each chunk to be absolute (from video start)."""
    if not chunks:
        return transcript

    result = transcript

    # Process each chunk
    for chunk in chunks:
        chunk_offset = chunk['start']  # Offset in seconds

        # Find the chunk section
        chunk_header = f"--- Chunk {chunk['chunk']} ({chunk['start_str']} - {chunk['end_str']}) ---"

        # Pattern for timestamps within this chunk: [MM:SS] or [HH:MM:SS]
        timestamp_pattern = r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]'

        # Split by chunk headers to process each section
        # We'll do this more carefully

    # Alternative approach: process line by line
    lines = transcript.split('\n')
    current_chunk_offset = 0
    new_lines = []

    for line in lines:
        # Check if this is a chunk header
        chunk_match = re.match(r'--- Chunk (\d+) \(([^)]+)\) ---', line)
        if chunk_match:
            # Extract chunk start time as offset
            time_range = chunk_match.group(2)
            start_str = time_range.split(' - ')[0]
            current_chunk_offset = parse_time(start_str)
            # Skip the chunk header line
            continue

        # Adjust timestamps in this line
        def adjust_match(m):
            original_time = parse_time(m.group(1))
            absolute_time = original_time + current_chunk_offset
            return f"[{format_time(absolute_time)}]"

        adjusted_line = re.sub(r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]', adjust_match, line)
        new_lines.append(adjusted_line)

    return '\n'.join(new_lines)


def cleanup_transcript_with_gemini(
    transcript: str,
    api_key: str,
    model_name: str = "gemini-2.0-flash",
    verbose: bool = True
) -> str:
    """Use Gemini to clean up the transcript in chunks."""

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # Split transcript into chunks by speaker turns (lines starting with [timestamp])
    lines = transcript.split('\n')

    # Group lines into speaker entries (entry starts with [timestamp])
    entries = []
    current_entry = []
    for line in lines:
        if re.match(r'\[\d{1,2}:\d{2}(:\d{2})?\]', line.strip()) and current_entry:
            entries.append('\n'.join(current_entry))
            current_entry = [line]
        else:
            current_entry.append(line)
    if current_entry:
        entries.append('\n'.join(current_entry))

    # Process in chunks of 15 entries (~15 speaker turns per chunk)
    chunk_size = 15
    cleaned_chunks = []

    for i in range(0, len(entries), chunk_size):
        chunk_entries = entries[i:i + chunk_size]
        chunk_text = '\n\n'.join(chunk_entries)

        if verbose:
            chunk_num = i // chunk_size + 1
            total_chunks = (len(entries) + chunk_size - 1) // chunk_size
            print(f"  Processing chunk {chunk_num}/{total_chunks}...")

        cleaned = cleanup_single_chunk(model, chunk_text, verbose=False)
        cleaned_chunks.append(cleaned)

    return '\n\n'.join(cleaned_chunks)


def cleanup_single_chunk(model, chunk_text: str, verbose: bool = True) -> str:
    """Clean up a single chunk of transcript."""

    prompt = """You are a transcript editor. Clean up the following transcript segment from a biotech/medical webinar.

IMPORTANT INSTRUCTIONS:

1. **PRESERVE ALL TIMESTAMPS** - Keep every [MM:SS] or [HH:MM:SS] timestamp exactly as they appear. These are critical for syncing with slides.

2. **Identify and name the speakers** based on context clues in the transcript:
   - The host introduces himself and the speakers at the beginning
   - Replace generic labels (SPEAKER_1, SPEAKER_2, etc.) with actual names
   - Based on the transcript, the speakers are:
     * Josh (the host/interviewer from a biotech analyst firm)
     * Kartik Ramamurthy (CEO, co-founder of Encoded Therapeutics)
     * Stephanie Tagliatella (co-founder and Chief Scientific Officer)
     * Salvador Rico (Chief Medical Officer)
   - Use first names: "JOSH:", "KARTIK:", "STEPHANIE:", "SALVADOR:"

3. **Fix transcription errors**:
   - Correct misspellings of technical/medical terms (e.g., "Drave" should be "Dravet", "gic" should be "GABAergic", "violent" should be "Vineland")
   - Fix obvious typos and repeated words
   - Correct company/drug names (ETX101 should be "ETX-101", SCN1A, NAV 1.1, etc.)
   - Fix any clearly wrong homophones

4. **Clean up formatting**:
   - Remove any "um", "uh" filler words that remain
   - Ensure consistent spacing
   - Keep paragraph breaks between different speaker turns

5. **DO NOT**:
   - Change the meaning or content
   - Remove or modify timestamps
   - Add new information
   - Summarize or shorten the content

Return the cleaned transcript segment maintaining the exact same structure with timestamps and speaker labels.

TRANSCRIPT SEGMENT TO CLEAN:
---
""" + chunk_text + """
---

Return the cleaned transcript segment:"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=30000,
            ),
            request_options={"timeout": 300}
        )

        if response.text:
            return response.text.strip()
        else:
            return chunk_text

    except Exception as e:
        print(f"Error cleaning chunk: {e}")
        return chunk_text


def main():
    parser = argparse.ArgumentParser(
        description="Clean up a transcript using Gemini API"
    )

    parser.add_argument("transcript", help="Path to the transcript file")
    parser.add_argument(
        "--output", "-o",
        help="Output file for cleaned transcript (default: transcript_cleaned.txt)"
    )
    parser.add_argument(
        "--api-key", "-k",
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Gemini API key required. Use --api-key or set GEMINI_API_KEY env var")
        sys.exit(1)

    # Read transcript
    transcript_path = Path(args.transcript).resolve()
    if not transcript_path.exists():
        print(f"Error: Transcript file not found: {transcript_path}")
        sys.exit(1)

    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = f.read()

    verbose = not args.quiet

    if verbose:
        print(f"Read transcript: {len(transcript)} characters")

    # Parse chunk info and adjust timestamps to absolute
    chunks = parse_chunk_info(transcript)
    if chunks:
        if verbose:
            print(f"Found {len(chunks)} chunks, adjusting timestamps to absolute...")
        transcript = adjust_timestamps(transcript, chunks)

    # Clean up with Gemini
    cleaned = cleanup_transcript_with_gemini(transcript, api_key, verbose=verbose)

    # Determine output path
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = transcript_path.parent / f"{transcript_path.stem}_cleaned.txt"

    # Save cleaned transcript
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)

    if verbose:
        print(f"\nCleaned transcript saved to: {output_path}")
        print(f"Length: {len(cleaned)} characters")


if __name__ == "__main__":
    main()
