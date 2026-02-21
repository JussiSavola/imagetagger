# ImageTagger

A Python CLI tool that uses Venice.ai vision models to automatically analyze images and embed AI-generated keywords into EXIF metadata. The tool has been observed working against OpenAI API and suitable vision models as well: Just set up url and api_key in the env.txt file.

These are good parameters with Venice API:
api_base=https://api.venice.ai/api/v1
model=google-gemma-3-27b-it

These work with OpenAI API:
api_base=https://api.openai.com/v1
model=gpt-4o-mini

Other providers probably work just as well.

## Features

- **AI-Powered Keyword Extraction**: Uses vision models to identify objects, scenes, colors, brands, and activities
- **EXIF Metadata Embedding**: Writes keywords directly into image metadata (XPKeywords, ImageDescription, UserComment)
- **Flexible Output**: Create enriched copies or overwrite originals in-place
- **Original File Safety**: Atomic writes for overwrite mode to prevent data corruption
- **Rate Limit Handling**: Exponential backoff for API rate limits
- **Balance Monitoring**: Real-time Venice.ai balance tracking
- **Verbose Mode**: Detailed debugging output for troubleshooting
- **Prevent Reprocessing**: Uses a configurable marker tag to avoid reprocessing already handled images
- **Custom Tag Support**: Set a custom marker to identify processed images
- **Marker Field Selection**: Store marker in XPComment field (-x) or in XPKeywords (default)
- **External Config**: Specify environment file location (useful for packaged executables)

## Requirements

- Python 3.9+
- Venice.ai API key

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
api_key=your-venice-api-key-here

# Optional (defaults shown)
api_base=https://api.venice.ai/api/v1
model=google-gemma-3-27b-it
```

Or specify a custom location with `-e` flag.

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
│   Input Image   │ ──▶ │  Venice.ai API  │ ──▶ │ Enriched Image  │
│                 │     │  Vision Model   │     │ + EXIF Keywords │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

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

The tool embeds a marker tag (default: "jms") to avoid reprocessing:
- **Default (no `-x`)**: Marker added to XPKeywords as part of keyword list
- **With `-x`**: Marker stored ONLY in XPComment field (not in keywords)

On subsequent runs, images containing this marker in the corresponding field are automatically skipped unless `-f` (force) is used.

### EXIF Fields Written

| Field | EXIF Tag | Content |
|-------|----------|---------|
| XPKeywords | 0x9C9E | Comma-separated AI keywords (Windows compatible) |
| XPSubject | 0x9C9F | Keywords duplicate |
| XPComment* | 0x9C9A | Processing marker (only with `-x`) |
| ImageDescription | 0x010E | AI analysis (truncated to 500 chars) |
| UserComment | 0x9286 | Full AI response |

\* Marker only written if `-x` flag is used. Without `-x`, marker is appended to XPKeywords.

## API Best Practices

This tool implements Venice.ai recommended practices:

- **Rate Limiting**: Monitors `x-ratelimit-remaining-*` headers with exponential backoff
- **Balance Tracking**: Displays `x-venice-balance-usd` after each request
- **Request Logging**: Captures `CF-RAY` header for support tickets
- **Deprecation Warnings**: Alerts on `x-venice-model-deprecation-warning`

## Example Output

### Standard Mode (marker in XPKeywords)

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

----------------------------------------------------------------------
[1/18] DSCF1234.JPG
----------------------------------------------------------------------

📋 EXISTING METADATA
  Format: JPEG | Size: 3840x2160
  [META] Make: Apple
  [META] DateTime: 2024:12:22 16:52:59

🤖 EXTRACTING KEYWORDS
    💰 Balance: $15.2347
  ✅ Keywords: car tire, snow chains, winter, snow...
     (12 total)

💾 SAVING
  ✅ Enriched: DSCF1234_enriched.JPG
  ✅ Keywords embedded in EXIF (marker 'jms' in XPKeywords)
  📝 Log: enriched\DSCF1234_keywords.txt

----------------------------------------------------------------------
[2/18] DSCF5678.JPG
----------------------------------------------------------------------
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

### XPComment Mode (`-x` flag)

```
Marker field: XPComment
✅ Keywords embedded in EXIF (marker 'jms' in XPComment)
```

### Verbose Mode (`-v`)

Shows API headers, all EXIF tags, raw AI responses, and debug information.

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

The resulting `imagetagger.exe` will need `-e` flag to locate the env.txt file unless you bundle it or place env.txt in the same directory as the executable.

## License

MIT License
