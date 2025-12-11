#!/usr/bin/env python3
"""
Transcript Generator using Gemini 2.5 Flash API

Extracts audio from video and generates a transcript with speaker diarization.

Usage:
    python generate_transcript.py <video_file> --api-key <GEMINI_API_KEY>

    # Or set API key as environment variable:
    export GEMINI_API_KEY=your_api_key
    python generate_transcript.py <video_file>
"""

import argparse
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple
import json

# Load .env file if present
try:
    from dotenv import load_dotenv
    # Look for .env in the script's directory
    script_dir = Path(__file__).parent
    env_path = script_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not available, use environment variables directly

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai not installed. Run: pip install google-generativeai")
    sys.exit(1)


def extract_audio_with_moviepy(video_path: str, output_path: str, verbose: bool = True) -> bool:
    """Extract audio from video using moviepy."""
    try:
        from moviepy import VideoFileClip

        if verbose:
            print(f"Extracting audio from video...")

        video = VideoFileClip(video_path)
        audio = video.audio

        if audio is None:
            print("Error: Video has no audio track")
            return False

        # Export as mp3 for smaller file size
        audio.write_audiofile(output_path)
        video.close()

        if verbose:
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"Audio extracted: {output_path} ({file_size_mb:.1f} MB)")

        return True
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False


def upload_to_gemini(file_path: str, mime_type: str, verbose: bool = True) -> Optional[object]:
    """Upload a file to Gemini's File API."""
    try:
        if verbose:
            print(f"Uploading file to Gemini...")

        file = genai.upload_file(file_path, mime_type=mime_type)

        # Wait for file to be processed
        while file.state.name == "PROCESSING":
            if verbose:
                print("  Processing...")
            time.sleep(5)
            file = genai.get_file(file.name)

        if file.state.name == "FAILED":
            print(f"Error: File processing failed")
            return None

        if verbose:
            print(f"File uploaded successfully: {file.name}")

        return file
    except Exception as e:
        print(f"Error uploading file: {e}")
        return None


def generate_transcript_gemini(
    audio_file: object,
    model_name: str = "gemini-2.0-flash",
    verbose: bool = True
) -> Optional[str]:
    """Generate transcript using Gemini API."""
    try:
        model = genai.GenerativeModel(model_name)

        prompt = """Please transcribe this audio recording. This is a business/medical presentation
with multiple speakers.

Instructions:
1. Identify different speakers and label them (Speaker 1, Speaker 2, etc., or by name if mentioned)
2. Include timestamps at the start of each speaker's turn (format: [MM:SS] or [HH:MM:SS])
3. Transcribe all spoken content accurately
4. Preserve natural speech patterns but clean up filler words (um, uh) for readability
5. Use paragraph breaks when speakers change or topics shift
6. If a speaker's name is mentioned, use their name instead of "Speaker N"

Format the output as:
[TIMESTAMP] SPEAKER_NAME: Transcribed text here.

Begin transcription:"""

        if verbose:
            print("Generating transcript with Gemini...")
            print("  This may take several minutes for long recordings...")

        response = model.generate_content(
            [audio_file, prompt],
            generation_config=genai.GenerationConfig(
                temperature=0.1,  # Low temperature for accuracy
                max_output_tokens=100000,  # Allow long output
            ),
            request_options={"timeout": 600}  # 10 minute timeout
        )

        if response.text:
            if verbose:
                print("Transcript generated successfully!")
            return response.text
        else:
            print("Error: Empty response from Gemini")
            return None

    except Exception as e:
        print(f"Error generating transcript: {e}")
        return None


def chunk_audio_for_long_videos(
    video_path: str,
    chunk_duration_minutes: int = 30,
    verbose: bool = True
) -> List[Tuple[str, float, float]]:
    """
    For very long videos, split audio into chunks.
    Returns list of (chunk_path, start_time, end_time).
    """
    try:
        from moviepy import VideoFileClip

        video = VideoFileClip(video_path)
        duration = video.duration
        video.close()

        chunk_duration_sec = chunk_duration_minutes * 60
        chunks = []

        if duration <= chunk_duration_sec:
            # Single chunk
            temp_path = tempfile.mktemp(suffix=".mp3")
            if extract_audio_with_moviepy(video_path, temp_path, verbose):
                chunks.append((temp_path, 0, duration))
        else:
            # Multiple chunks
            num_chunks = int(duration / chunk_duration_sec) + 1
            if verbose:
                print(f"Video is {duration/60:.1f} minutes, splitting into {num_chunks} chunks...")

            for i in range(num_chunks):
                start = i * chunk_duration_sec
                end = min((i + 1) * chunk_duration_sec, duration)

                if start >= duration:
                    break

                temp_path = tempfile.mktemp(suffix=f"_chunk{i+1}.mp3")

                if verbose:
                    print(f"  Extracting chunk {i+1}: {start/60:.1f} - {end/60:.1f} minutes")

                # Reload video for each chunk to avoid internal state issues
                video = VideoFileClip(video_path)
                audio_clip = video.audio.subclipped(start, end)
                audio_clip.write_audiofile(temp_path)
                audio_clip.close()
                video.close()

                chunks.append((temp_path, start, end))

        return chunks

    except Exception as e:
        print(f"Error chunking audio: {e}")
        return []


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def generate_full_transcript(
    video_path: str,
    api_key: str,
    output_path: Optional[str] = None,
    chunk_duration_minutes: int = 30,
    verbose: bool = True
) -> Optional[str]:
    """
    Generate full transcript for a video file.

    Args:
        video_path: Path to video file
        api_key: Gemini API key
        output_path: Optional path to save transcript
        chunk_duration_minutes: Max duration per chunk for long videos
        verbose: Print progress

    Returns:
        Full transcript text or None on error
    """
    start_time = time.time()

    # Configure Gemini
    genai.configure(api_key=api_key)

    # Get video duration
    try:
        from moviepy import VideoFileClip
        video = VideoFileClip(video_path)
        duration = video.duration
        video.close()

        if verbose:
            print(f"Video duration: {duration/60:.1f} minutes")
    except Exception as e:
        print(f"Error reading video: {e}")
        return None

    # Extract and process audio
    if duration <= chunk_duration_minutes * 60:
        # Single chunk processing
        audio_path = tempfile.mktemp(suffix=".mp3")

        if not extract_audio_with_moviepy(video_path, audio_path, verbose):
            return None

        try:
            # Upload to Gemini
            audio_file = upload_to_gemini(audio_path, "audio/mpeg", verbose)
            if audio_file is None:
                return None

            # Generate transcript
            transcript = generate_transcript_gemini(audio_file, verbose=verbose)

            # Clean up uploaded file
            try:
                genai.delete_file(audio_file.name)
            except:
                pass

        finally:
            # Clean up temp file
            if os.path.exists(audio_path):
                os.remove(audio_path)
    else:
        # Multi-chunk processing for long videos
        chunks = chunk_audio_for_long_videos(video_path, chunk_duration_minutes, verbose)

        if not chunks:
            return None

        transcript_parts = []

        for i, (chunk_path, start, end) in enumerate(chunks):
            if verbose:
                print(f"\nProcessing chunk {i+1}/{len(chunks)} ({format_timestamp(start)} - {format_timestamp(end)})")

            try:
                audio_file = upload_to_gemini(chunk_path, "audio/mpeg", verbose)
                if audio_file is None:
                    continue

                chunk_transcript = generate_transcript_gemini(audio_file, verbose=verbose)

                if chunk_transcript:
                    # Add time offset note
                    header = f"\n--- Chunk {i+1} ({format_timestamp(start)} - {format_timestamp(end)}) ---\n"
                    transcript_parts.append(header + chunk_transcript)

                # Clean up
                try:
                    genai.delete_file(audio_file.name)
                except:
                    pass

            finally:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)

        transcript = "\n".join(transcript_parts)

    # Save transcript if output path provided
    if output_path and transcript:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        if verbose:
            print(f"\nTranscript saved to: {output_path}")

    # Report timing
    elapsed = time.time() - start_time
    if verbose:
        print(f"\nTotal processing time: {elapsed/60:.1f} minutes")

    return transcript


def main():
    parser = argparse.ArgumentParser(
        description="Generate transcript from video using Gemini API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_transcript.py video.mp4 --api-key YOUR_API_KEY
  python generate_transcript.py video.mp4 -o transcript.txt

Environment variable:
  export GEMINI_API_KEY=your_api_key
  python generate_transcript.py video.mp4
        """
    )

    parser.add_argument("video", help="Path to the video file")
    parser.add_argument(
        "--api-key", "-k",
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for transcript (default: video_name_transcript.txt)"
    )
    parser.add_argument(
        "--chunk-duration", "-c",
        type=int,
        default=30,
        help="Max chunk duration in minutes for long videos (default: 30)"
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

    # Validate input
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = video_path.parent / f"{video_path.stem}_transcript.txt"

    # Generate transcript
    transcript = generate_full_transcript(
        str(video_path),
        api_key,
        str(output_path),
        chunk_duration_minutes=args.chunk_duration,
        verbose=not args.quiet
    )

    if transcript is None:
        print("Error: Failed to generate transcript")
        sys.exit(1)

    print(f"\nTranscript length: {len(transcript)} characters")


if __name__ == "__main__":
    main()
