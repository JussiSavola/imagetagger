
### README.md

```markdown
# Image Metadata Enricher

A Python CLI tool that uses Venice.ai vision models to automatically analyze images and embed AI-generated keywords into EXIF metadata.

## Features

- **AI-Powered Keyword Extraction**: Uses vision models to identify objects, scenes, colors, brands, and activities
- **EXIF Metadata Embedding**: Writes keywords directly into image metadata (XPKeywords, ImageDescription, UserComment)
- **Flexible Output**: Create enriched copies or overwrite originals in-place
- **Original File Safety**: Atomic writes for overwrite mode to prevent data corruption
- **Rate Limit Handling**: Exponential backoff for API rate limits
- **Balance Monitoring**: Real-time Venice.ai balance tracking
- **Verbose Mode**: Detailed debugging output for troubleshooting

## Requirements

- Python 3.9+
- Venice.ai API key

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/image-metadata-enricher.git
cd image-metadata-enricher

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
Pillow>=10.0.0
requests>=2.31.0
piexif>=1.1.3
tqdm>=4.66.0
```

## Configuration

Create an `env.txt` file in the project directory:

```text
# Required
api_key=your-venice-api-key-here

# Optional (defaults shown)
api_base=https://api.venice.ai/api/v1
model=google-gemma-3-27b-it
```

## Usage

```
usage: imagetagger.py [-h] [-d DIRECTORY] [-o] [-v]

Enrich image metadata using Venice.ai vision models.

options:
  -h, --help            show this help message and exit
  -d DIRECTORY, --directory DIRECTORY
                        Directory containing images (default: current directory)
  -o, --overwrite       Overwrite original images instead of creating enriched copies
  -v, --verbose         Print detailed debug information
```

### Examples

```bash
# Process current directory, save copies to ./enriched/
python imagetagger.py

# Process specific directory
python imagetagger.py -d /path/to/photos

# Overwrite original images (use with caution)
python imagetagger.py -d ./photos -o

# Verbose mode for debugging
python imagetagger.py -v

# Combine all options
python imagetagger.py -d ./vacation-photos -o -v
```

## How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Input Image   │ ──▶ │  Venice.ai API  │ ──▶ │ Enriched Image  │
│                 │     │  Vision Model   │     │ + EXIF Keywords │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Output Modes

| Mode | Command | Behavior |
|------|---------|----------|
| **Copy** (default) | `python imagetagger.py` | Creates `enriched/` subfolder with `_enriched` suffix |
| **Overwrite** | `python imagetagger.py -o` | Modifies original files in-place using atomic writes |

### EXIF Fields Written

| Field | EXIF Tag | Content |
|-------|----------|---------|
| XPKeywords | 0x9C9E | Comma-separated keywords (Windows compatible) |
| XPSubject | 0x9C9F | Keywords duplicate |
| ImageDescription | 0x010E | AI analysis (truncated to 500 chars) |
| UserComment | 0x9286 | Full AI response |

## API Best Practices

This tool implements Venice.ai recommended practices:

- **Rate Limiting**: Monitors `x-ratelimit-remaining-*` headers with exponential backoff
- **Balance Tracking**: Displays `x-venice-balance-usd` after each request
- **Request Logging**: Captures `CF-RAY` header for support tickets
- **Deprecation Warnings**: Alerts on `x-venice-model-deprecation-warning`

## Example Output

### Standard Mode

```
======================================================================
IMAGE KEYWORD EXTRACTOR - Venice.ai
======================================================================
Model: google-gemma-3-27b-it
Input:  /Users/user/photos
Output: /Users/user/photos/enriched
======================================================================

Found 18 image(s)

──────────────────────────────────────────────────────────────────────
[1/18] DSCF1234.JPG
──────────────────────────────────────────────────────────────────────

📋 EXISTING METADATA
  Format: JPEG | Size: 3840x2160
  [META] Make: Apple
  [META] DateTime: 2024:12:22 16:52:59

🤖 EXTRACTING KEYWORDS
    💰 Balance: $15.2347
  ✅ Keywords: car tire, snow chains, winter, snow...
     (12 total)

💾 SAVING
  Enriched image: DSCF1234_enriched.JPG
  ✓ Keywords embedded in EXIF
```

### Verbose Mode (`-v`)

```
📋 EXISTING METADATA
  Format: JPEG | Size: 3840x2160
  Mode: RGB | Bits: 8
  [EXIF] Make: Apple
  [EXIF] Model: iPhone SE (3rd generation)
  [EXIF] Orientation: 1
  [EXIF] XResolution: 72
  ...

  [VERBOSE] API Request Payload:
    Model: google-gemma-3-27b-it
    Message Count: 2

  [VERBOSE] API Response Headers:
    x-ratelimit-remaining-requests: 29
    x-venice-balance-usd: 15.2334
    CF-RAY: 8x7abc123def-AMS

  [VERBOSE] Raw AI Response:
  car tire, snow chains, winter, snow, vehicle, transportation
```

## Troubleshooting

### "Model not found" (404)

Verify the model name in your `env.txt`:
```bash
curl https://api.venice.ai/api/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Keywords not appearing in Windows Explorer

Windows caches thumbnail metadata. Use a tool like ExifTool to verify:
```bash
exiftool -XPKeywords image.jpg
```

## License

MIT License
```
