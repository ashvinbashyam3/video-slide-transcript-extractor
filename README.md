# Video Slide & Transcript Extractor

A Python application that extracts presentation slides and generates transcripts from video recordings (e.g., Zoom webinars, presentations). Produces a single self-contained HTML document combining slides and transcript.

## Features

- **Automatic Slide Detection**: Identifies and extracts unique presentation slides from video
- **AI-Powered Transcription**: Uses Google Gemini API for accurate speech-to-text with speaker diarization
- **Speaker Identification**: Attempts to identify speakers by name from context
- **Slide Captioning**: Generates descriptive captions for each slide using vision AI
- **Self-Contained Output**: Produces a single HTML file with embedded images (no external dependencies)
- **Windows Integration**: Includes batch file for easy one-click execution

## Quick Start

### Windows Users

1. Double-click `Extract Video Slides.bat`
2. Select your video file from the file dialog (opens to Downloads folder)
3. Wait for processing (typically 5-10 minutes for a 1-hour video)
4. Find the output HTML file in the same folder as your video

### Command Line

```bash
# Opens file dialog
python main.py

# Process specific file
python main.py path/to/video.mp4
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install opencv-python numpy moviepy google-generativeai python-dotenv
```

### Step 2: Configure API Key

Create a `.env` file in the project directory:

```
GEMINI_API_KEY=your_api_key_here
```

Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | >=4.5 | Video frame extraction and image processing |
| `numpy` | >=1.19 | Numerical computations for image analysis |
| `moviepy` | >=2.0 | Audio extraction from video |
| `google-generativeai` | >=0.3 | Gemini API for transcription and vision |
| `python-dotenv` | >=0.19 | Environment variable management |

### System Requirements

- **OS**: Windows 10/11, macOS, or Linux
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: Temporary space for audio processing (~500MB per hour of video)
- **Network**: Internet connection for Gemini API calls

## Output Format

The program generates a single HTML file with:

- Clean, responsive design
- Transcript with speaker labels
- Slides embedded as base64 images at their presentation timestamps
- AI-generated captions for each slide

Example output filename: `video_name.html` (same location as input video)

---

# Technical Documentation

## Architecture Overview

The program consists of four main processing stages:

```
Video File
    │
    ├─► [1] Slide Extraction (OpenCV)
    │       └─► Slides in memory (JPEG bytes)
    │
    ├─► [2] Audio Extraction (moviepy)
    │       └─► Audio chunks (MP3)
    │
    ├─► [3] Transcription (Gemini API)
    │       └─► Transcript entries
    │
    └─► [4] HTML Generation
            └─► Single HTML file with embedded slides
```

## Slide Detection Methodology

### Overview

The slide detection system uses a multi-stage approach combining computer vision techniques to identify when presentation slides appear and change in a video recording.

### Stage 1: Slide Region Detection

**Problem**: In video recordings (especially Zoom), the frame contains both the presentation slide and speaker video thumbnails. We need to identify where the slide is located.

**Solution**: Analyze frame layout to find the slide region.

```python
def find_slide_region(cap, num_samples=30):
    # 1. Sample 30 frames throughout the video
    # 2. Find frames with high white content (likely slides)
    # 3. Analyze column-wise white pixel distribution
    # 4. Find where white content drops (slide/speaker boundary)
    # 5. Snap to standard aspect ratio (16:9, 16:10, 4:3)
```

**Algorithm Details**:

1. **Frame Sampling**: Sample frames at 15-85% of video duration to avoid intro/outro
2. **White Content Detection**: Filter frames where >20% pixels have brightness >200
3. **Boundary Detection**:
   - Analyze top third of frame (where speaker thumbnails usually appear)
   - Count white pixels per column
   - Find rightmost column with >30% white pixels
4. **Aspect Ratio Snapping**: Match detected dimensions to nearest standard ratio

### Stage 2: Slide Visibility Detection

**Problem**: Not every frame with the slide region visible actually shows a slide. Sometimes only the speaker is visible.

**Solution**: Multi-factor analysis to distinguish slides from speaker-only views.

```python
def is_slide_visible(frame, region):
    # Check 4 factors:
    # 1. White/light background ratio (slides typically have white backgrounds)
    # 2. Edge density (text/graphics create specific edge patterns)
    # 3. Skin tone ratio (speaker views have more skin-colored pixels)
    # 4. Content characteristics
```

**Detection Criteria**:

| Factor | Slide Threshold | Rationale |
|--------|-----------------|-----------|
| White pixels (>200) | >25% | Slides typically have white backgrounds |
| Light pixels (>180) | >40% | Alternative for light-themed slides |
| Skin tone pixels | <12% | Speaker faces have skin colors |
| Edge density | 1.5-20% | Text/graphics have moderate edges |

**Skin Tone Detection** (HSV color space):
- Hue: 0-25 (orange-red range)
- Saturation: 40-180 (not grayscale, not oversaturated)
- Value: >80 (not too dark)

### Stage 3: Slide Change Detection

**Problem**: Identify when one slide transitions to another.

**Solution**: Perceptual hashing with temporal analysis.

#### Perceptual Hash Algorithm (pHash)

```python
def compute_phash(image, hash_size=16):
    # 1. Resize to 16x16
    # 2. Convert to grayscale
    # 3. Apply DCT (Discrete Cosine Transform)
    # 4. Keep top-left 8x8 (low frequencies)
    # 5. Compute median
    # 6. Create binary hash (above/below median)
```

**Why DCT-based pHash?**
- Robust to compression artifacts (video encoding)
- Ignores fine details, captures overall structure
- Fast to compute
- Produces fixed-size hash (64 bits)

#### Change Detection Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `hash_threshold` | 0.15 | Hamming distance threshold for change detection |
| `sample_interval` | 2.0s | How often to check for changes |
| `min_slide_duration` | 3.0s | Minimum time between slides |
| `transition_delay` | 500ms | Wait after change for transition to complete |

#### Duplicate Slide Filtering

To avoid capturing the same slide multiple times (e.g., returning to a previous slide):
- Maintain list of all captured slide hashes
- Before saving new slide, check against all previous slides
- Skip if Hamming distance < threshold to any previous slide

### Stage 4: Transition Handling

**Problem**: Slide transitions (fades, wipes) create intermediate frames that aren't the actual slide.

**Solution**: Stability detection after change.

```python
# After detecting a change:
# 1. Wait 500ms for transition to start completing
# 2. Read frames and compare consecutive hashes
# 3. When consecutive frames have hash distance < 0.05, transition is complete
# 4. Capture the stable frame
```

## Transcript Generation Methodology

### Audio Extraction

**Library**: moviepy (Python wrapper for FFmpeg)

**Process**:
1. Load video file
2. Extract audio track
3. For long videos (>20 min), split into 20-minute chunks
4. Export as MP3 (128kbps) for efficient upload

**Why chunk audio?**
- Gemini has processing limits for large files
- Parallel processing potential
- Memory efficiency
- Better error recovery

### Gemini Speech-to-Text

**API**: Google Gemini 2.0 Flash

**Process**:
1. Upload audio chunk to Gemini File API
2. Wait for server-side processing (PROCESSING → ACTIVE)
3. Send transcription prompt with speaker diarization instructions
4. Parse response into structured entries

**Prompt Design**:
```
Transcribe this audio with speaker diarization.
- Identify different speakers as SPEAKER_1, SPEAKER_2, etc.
- Format: SPEAKER_X: [their speech]
- Include all spoken content
- One speaker turn per line
```

### Transcript Cleanup

**Process**:
1. Group entries into chunks of 15
2. Send to Gemini with cleanup prompt
3. Replace generic speaker labels with actual names (if identifiable)
4. Fix transcription errors, especially technical terms

**Challenges with Cleanup**:
- Large transcripts can be truncated by the model
- Solution: Process in small chunks (15 entries at a time)
- Maintain context across chunks for consistent speaker naming

## Slide Captioning

**API**: Gemini 2.0 Flash (Vision)

**Process**:
1. Send slide image (JPEG bytes) to Gemini
2. Request 2-3 sentence description
3. Focus on main topic and key data points

## HTML Generation

### Slide Placement Algorithm

Slides are placed at paragraph boundaries based on their presentation timestamps:

```python
# For each slide:
# 1. Get slide timestamp (when it appeared in video)
# 2. Estimate corresponding transcript position
# 3. Insert slide after that transcript entry
```

**Formula**:
```python
entries_per_sec = len(transcript) / max_slide_time
entry_index = int(slide_timestamp * entries_per_sec)
```

### Image Embedding

Images are embedded as base64 data URIs:
- No external file dependencies
- Single self-contained HTML file
- Works offline after generation
- Larger file size (~1.5x original image sizes)

## Challenges Encountered

### 1. moviepy API Changes (v2.x)

**Problem**: moviepy v2.2.1 removed `verbose` and `logger` parameters from `write_audiofile()`.

**Solution**: Removed these parameters from all calls.

### 2. VideoFileClip State Issues

**Problem**: Reusing a single VideoFileClip object across multiple chunk extractions caused internal state corruption.

**Error**: `'NoneType' object has no attribute 'stdout'`

**Solution**: Create fresh VideoFileClip instance for each chunk extraction:
```python
# Inside loop for each chunk:
video = VideoFileClip(video_path)
audio_clip = video.audio.subclipped(start, end)
# ... process ...
video.close()
```

### 3. Transcript Truncation

**Problem**: Sending full 60K+ character transcript to Gemini resulted in truncated output (~38K chars).

**Solution**:
- Split transcript by speaker turns (detect `[timestamp]` pattern)
- Process 15 speaker turns per API call
- Combine results

**Key Insight**: The transcript uses single `\n` between lines, not `\n\n`, so splitting by `\n\n` failed.

### 4. Timestamp Confusion at Chunk Boundaries

**Problem**: When transcribing long videos in chunks, timestamps reset to 00:00 at each chunk boundary.

**Solution**:
- Track chunk start time offset
- Adjust all timestamps within each chunk to absolute time

### 5. Slide Region Detection in Various Layouts

**Problem**: Different video conferencing tools have different layouts.

**Solution**:
- Multiple detection strategies
- Fallback to 70% width assumption
- Snap to standard aspect ratios for clean crops

### 6. False Positive Slide Detection

**Problem**: Some speaker views with uniform backgrounds were detected as slides.

**Solution**: Added stricter criteria:
- Require actual white/light background (not just uniform color)
- Check edge density range (too many edges = face, too few = blank)
- Skin tone filtering

## File Structure

```
video-slide-extractor/
├── main.py                    # Main unified program
├── Extract Video Slides.bat   # Windows launcher
├── requirements.txt           # Python dependencies
├── .env                       # API key (create this, not in repo)
├── .gitignore                 # Git ignore rules
├── README.md                  # This documentation
│
└── [Legacy/Development files - can be removed]
    ├── extract_slides.py      # Standalone slide extractor
    ├── generate_transcript.py # Standalone transcript generator
    ├── cleanup_transcript.py  # Transcript cleanup utility
    └── create_html_report.py  # HTML report generator
```

## API Reference

### Google Gemini API

**Endpoints Used**:

1. **File Upload API**: `genai.upload_file()`
   - Upload audio files for transcription
   - Max file size: 2GB
   - Supports: MP3, WAV, FLAC, etc.

2. **Generate Content API**: `model.generate_content()`
   - Text generation for transcription
   - Vision analysis for slide captioning
   - Model: `gemini-2.0-flash`

**Rate Limits** (Free tier):
- 15 requests per minute
- 1 million tokens per minute
- 1500 requests per day

**Cost** (Pay-as-you-go):
- Input: $0.075 per 1M tokens
- Output: $0.30 per 1M tokens
- Audio: $0.00025 per second

### OpenCV Functions Used

| Function | Purpose |
|----------|---------|
| `cv2.VideoCapture()` | Open video file |
| `cv2.resize()` | Resize images for hashing |
| `cv2.cvtColor()` | Color space conversion |
| `cv2.dct()` | Discrete Cosine Transform |
| `cv2.Canny()` | Edge detection |
| `cv2.imencode()` | Encode image to JPEG bytes |

## Performance Characteristics

| Video Duration | Slides | Processing Time* | Output Size |
|----------------|--------|------------------|-------------|
| 15 minutes | ~5 | 2-3 min | ~1 MB |
| 30 minutes | ~10 | 4-5 min | ~2 MB |
| 60 minutes | ~16 | 8-12 min | ~5 MB |

*Depends on internet speed and API response times.

## Troubleshooting

### "GEMINI_API_KEY not found"
- Create `.env` file with your API key
- Or set environment variable: `set GEMINI_API_KEY=your_key`

### "Could not open video file"
- Ensure file path is correct
- Check file isn't corrupted
- Try a different video format

### Transcript is empty or incomplete
- Check API key has sufficient quota
- Video may have poor audio quality
- Try processing shorter sections

### No slides detected
- Video may not have typical slide appearance
- Adjust `hash_threshold` parameter (lower = more sensitive)
- Slides may not have white backgrounds

## Future Improvements

1. **Local Speech-to-Text**: Integrate Whisper for offline transcription
2. **Custom Slide Detection**: Train model for specific presentation styles
3. **Progress UI**: Add graphical progress indicator
4. **Parallel Processing**: Process audio and slides simultaneously
5. **Incremental Output**: Generate HTML progressively

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Google Gemini API for AI capabilities
- OpenCV community for computer vision tools
- moviepy developers for video processing
