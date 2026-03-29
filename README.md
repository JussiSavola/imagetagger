# ImageTagger

A Python CLI tool that uses vision AI models to automatically analyze images and embed AI-generated keywords into EXIF metadata. Works with Venice.ai, OpenAI, and any compatible API provider.

These are good parameters with Venice API:
```
api_base=https://api.venice.ai/api/v1
model=google-gemma-3-27b-it
```

These work with OpenAI API:
```
api_base=https://api.openai.com/v1
model=gpt-4o-mini
```

Other OpenAI-compatible providers work too.

## Features

- **AI-Powered Keyword Extraction**: Uses vision models to identify objects, scenes, colors, brands, and activities
- **Model Name in Keywords**: The model used is automatically appended to the keyword list on save
- **EXIF Metadata Embedding**: Writes keywords directly into image metadata (XPKeywords, XPSubject, ImageDescription, UserComment)
- **Flexible Output**: Create enriched copies or overwrite originals in-place
- **Original File Safety**: Atomic writes for overwrite mode to prevent data corruption
- **Keyword Log**: Saves a `_keywords.txt` log to the `enriched/` folder for every processed image
- **Rate Limit Handling**: Adaptive throttle delay (increases on rate limit, decreases on success) plus API-directed wait times
- **Balance Monitoring**: Real-time Venice.ai balance tracking with three tiers — warn at <$1.00, slow down (60s wait) at <$0.50, abort at <$0.10
- **Abort on Invalid Key**: Immediately stops processing on a 401 Invalid API Key response
- **Content Refusal Detection**: Skips images where the model refuses to analyze (detects "sorry", "can't", "cannot")
- **Thinking Block Stripping**: Removes `<think>...</think>` reasoning blocks from model output before parsing
- **Verbose Mode**: Detailed debugging output for troubleshooting
- **Prevent Reprocessing**: Uses a configurable marker tag to avoid reprocessing already handled images
- **Custom Tag Support**: Set a custom marker to identify processed images
- **Marker Field Selection**: Store marker in XPComment field (`-x`) or in XPKeywords (default)
- **External Config**: Specify environment file location (useful for packaged executables)

## Requirements

- Python 3.9+
- An API key for Venice.ai, OpenAI, or a compatible provider

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/imagetagger.git
cd imagetagger

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
requirements.txt:

Pillow>=10.0.0
requests>=2.31.0
piexif>=1.1.3
```

## Configuration

Create an `env.txt` file (by default in the same directory as the script):

```text
# Required
api_key=your-api-key-here

# Optional (defaults shown)
api_base=https://api.venice.ai/api/v1
model=google-gemma-3-27b-it
```

Or specify a custom location with the `-e` flag.

## Usage

```
usage: imagetagger [-h] [-d DIRECTORY] [-e ENV] [-o] [-v] [-f] [-t TAG] [-x]

Tag images with AI-generated keywords using Venice.ai vision models.

options:
  -h, --help            show this help message and exit
  -d DIRECTORY, --directory DIRECTORY
                        Directory containing images (default: current directory)
  -e ENV, --env ENV     Path to environment file with API key (default: script
                        directory/env.txt)
  -o, --overwrite       Overwrite original images instead of creating copies
  -v, --verbose         Enable verbose output (API details, all EXIF tags, etc.)
  -f, --force           Force reprocessing of all images, even if already processed
  -t TAG, --tag TAG     Custom marker tag for processed images (default: jms)
  -x, --xpcomment       Store marker ONLY in XPComment field (not in keywords list)
```

### Examples

```bash
# Process current directory, save copies to ./enriched/
python imagetagger.py

# Process specific directory with custom env file
python imagetagger.py -d ./photos -e C:\config\venice-env.txt

# Overwrite originals with custom marker (in keywords list)
python imagetagger.py -d ./photos -o -t "ai-processed"

# Store marker in XPComment field (separate from keywords)
python imagetagger.py -d ./photos -x -t "photoapp"

# Force reprocess all (ignore existing markers)
python imagetagger.py -d ./photos -f

# Verbose mode for debugging
python imagetagger.py -d ./photos -v

# Full example: overwrite originals, custom marker, XPComment, force reprocess
python imagetagger.py -d ./photos -o -t "mytag" -x -f -v
```

## How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Input Image   │ ──▶ │  Vision AI API  │ ──▶ │ Enriched Image  │
│                 │     │  (Venice/OAI)   │     │ + EXIF Keywords │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

1. Image is resized to max 800×600 and base64-encoded for the API
2. The vision model returns a comma-separated keyword list
3. Keywords are written into EXIF fields; the model name is appended to the list
4. A text log of the keywords and raw AI response is saved to `enriched/`

### Marker Storage Modes

| Option | Marker Location | Pros | Cons |
|--------|-----------------|------|------|
| Default (`-t TAG`) | Added to XPKeywords keyword list | Simple, visible in most apps | Marker appears in keyword searches |
| With `-x` (`-t TAG -x`) | Stored in XPComment field | Clean keyword list, marker hidden | Not visible in all apps |

### Output Modes

| Mode | Command | Behavior |
|------|---------|----------|
| **Copy** (default) | `python imagetagger.py` | Creates `enriched/` subfolder with `_enriched` suffix |
| **Overwrite** | `python imagetagger.py -o` | Modifies original files in-place using atomic writes |

### Reprocessing Prevention

The tool embeds a marker tag (default: `jms`) to avoid reprocessing:
- **Default (no `-x`)**: Marker added to XPKeywords as part of keyword list
- **With `-x`**: Marker stored ONLY in XPComment field (not in keywords)

On subsequent runs, images containing this marker in the corresponding field are automatically skipped unless `-f` (force) is used.

### EXIF Fields Written

| Field | EXIF Tag | Content |
|-------|----------|---------|
| XPKeywords | 0x9C9E | Comma-separated AI keywords + model name (Windows compatible) |
| XPSubject | 0x9C9F | Same as XPKeywords |
| XPComment* | 0x9C9A | Processing marker (only with `-x`) |
| ImageDescription | 0x010E | AI analysis (truncated to 500 chars) |
| UserComment | 0x9286 | Full AI response |

\* Marker only written to XPComment if `-x` flag is used. Without `-x`, marker is appended to XPKeywords.

## Cost Examples

Using Venice.ai (as of 2026-03-29):

| Model | Cost per image | ~1000 images |
|-------|---------------|--------------|
| `google-gemma-3-27b-it` | $0.000050 (~0.005 ¢) | ~$0.05 |
| `qwen3-5-9b` | $0.000028 (~0.003 ¢) | ~$0.03 |
| `claude-sonnet-4-6` | $0.015 per image (~1.5 ¢) | ~$15 |

Processing a large batch is very cheap — 10 000 images costs roughly $0.30–$0.50 using low end models.

## API Best Practices

This tool implements Venice.ai recommended practices:

- **Adaptive Throttle**: Inter-request delay increases 10% on rate limit, decreases 2% on success (floor: 0.1s)
- **Rate Limit Headers**: Waits the API-specified reset duration from `x-ratelimit-reset-*` headers on 429 responses, falls back to exponential backoff
- **Balance Monitoring**: Displays and acts on `x-venice-balance-usd` after each request:
  - `< $1.00` — warns to top up soon
  - `< $0.50` — warns and inserts a 60-second wait before the next image
  - `< $0.10` — aborts immediately
- **Instant 401 Abort**: Stops processing and exits on an invalid API key
- **Request Logging**: Captures `CF-RAY` header for support tickets
- **Deprecation Warnings**: Alerts on `x-venice-model-deprecation-warning`
- **Venice Parameters**: Sends `disable_thinking`, `strip_thinking_response`, and `include_venice_system_prompt: false` when Venice is detected
- **Thinking Block Stripping**: Removes `<think>...</think>` sections from model responses before keyword parsing

## Example Output

```
======================================================================
IMAGETAGGER - Venice.ai Image Keyword Extractor
======================================================================
Model:  google-gemma-3-27b-it
Vision: Yes
Input:  C:\Users\name\photos
Mode:   Create copies in enriched subfolder
Marker: 'jms'
Marker field: XPKeywords (in keyword list)
======================================================================

Found 18 image(s)

──────────────────────────────────────────────────────────────────────
[1/18] DSCF1234.JPG
──────────────────────────────────────────────────────────────────────

📋 EXISTING METADATA
  Format: JPEG | Size: 3840x2160
  [META] Make: Apple
  [META] DateTime: 2024:12:22 16:52:59

🖼️  RESIZING
  3840x2160 → 800x450

🤖 EXTRACTING KEYWORDS
    💰 Balance: $15.2347
  ✅ Keywords: car tire, snow chains, winter, snow, road...
     (13 total)

💾 SAVING
  ✅ Enriched: DSCF1234_enriched.JPG
  ✅ Keywords embedded in EXIF (marker 'jms' in XPKeywords)
  📝 Log: enriched\DSCF1234_keywords.txt

──────────────────────────────────────────────────────────────────────
[2/18] DSCF5678.JPG
──────────────────────────────────────────────────────────────────────
  ⏭️  SKIPPED - Already processed ('jms' marker found)

...

======================================================================
COMPLETE
  Processed: 17
  Skipped:   1
  Errors:    0
  Total:     18
======================================================================
```

### Verbose Mode (`-v`)

Shows API headers, all EXIF tags, raw AI responses, written EXIF fields, and debug information.

## Troubleshooting

### "Config file not found"

When using PyInstaller or moving the executable, use `-e` to specify the env.txt location:
```bash
imagetagger.exe -d ./photos -e C:\Users\name\AppData\Local\venice-env.txt
```

### "Model not found" (404)

Verify available models:
```bash
curl https://api.venice.ai/api/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Checking marker location

Without `-x`:
```bash
exiftool -XPKeywords image.jpg
```

With `-x`:
```bash
exiftool -XPComment image.jpg
```

### Keywords not appearing in Windows Explorer

Windows caches thumbnail metadata. Use ExifTool to verify:
```bash
exiftool -XPKeywords image.jpg
```

### Preventing reprocessing not working

Ensure `piexif` is installed:
```bash
pip install piexif
```

## Packaging as Windows Executable

Using PyInstaller:

```bash
pip install pyinstaller
pyinstaller --onefile imagetagger.py
```

The resulting `imagetagger.exe` will need the `-e` flag to locate `env.txt` unless it is placed in the same directory as the executable.

## License

MIT License
