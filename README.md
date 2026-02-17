# Image Metadata Enricher

A Python tool that uses Venice.ai vision models to automatically analyze images and embed AI-generated keywords into EXIF metadata.

## Features

- **AI-Powered Keyword Extraction**: Uses vision models to identify objects, scenes, colors, brands, and activities
- **EXIF Metadata Embedding**: Writes keywords directly into image metadata (XPKeywords, ImageDescription, UserComment)
- **Original File Preservation**: Never modifies source images - creates enriched copies
- **Rate Limit Handling**: Exponential backoff for API rate limits
- **Balance Monitoring**: Real-time Venice.ai balance tracking
- **Comprehensive Logging**: CF-RAY headers captured for troubleshooting

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

### Supported Vision Models

| Model | Speed | Quality | Notes |
|-------|-------|---------|-------|
| `google-gemma-3-27b-it` | Fast | High | Recommended default |
| `llama-3.2-90b-vision-instruct` | Medium | Highest | Best accuracy |
| `llama-3.2-11b-vision-instruct` | Fastest | Good | Lightweight option |
| `qwen2.5-vl-32b-instruct` | Medium | High | Alternative option |

## Usage

### Basic Usage

```bash
python image-enricher.py
```

Processes all JPG/JPEG/PNG images in the current directory.

### Specify Directory

```python
from image_enricher import process_images

process_images("/path/to/images")
```

### Output Structure

```
your-images/
├── photo1.jpg
├── photo2.jpg
└── enriched/
    ├── photo1_enriched.jpg      # Copy with embedded keywords
    ├── photo1_keywords.txt      # AI analysis log
    ├── photo2_enriched.jpg
    └── photo2_keywords.txt
```

## How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Input Image   │ ──▶ │  Venice.ai API  │ ──▶ │ Enriched Image  │
│  (preserved)    │     │  Vision Model   │     │ + EXIF Keywords │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │
        ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  EXIF Metadata  │     │ Keyword List    │
│  extracted for  │     │ "car tire, snow │
│    context      │     │ chains, winter" │
└─────────────────┘     └─────────────────┘
```

### Process Flow

1. **Extract** existing EXIF metadata (camera, date, GPS)
2. **Resize** image to 800x600 for API optimization
3. **Analyze** with Venice.ai vision model
4. **Parse** comma-separated keywords from response
5. **Embed** keywords into EXIF fields
6. **Verify** metadata integrity

## EXIF Fields Written

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

### Console

```
======================================================================
IMAGE KEYWORD EXTRACTOR - Venice.ai (Production Ready)
======================================================================
Model: google-gemma-3-27b-it
Vision: Yes
Endpoint: https://api.venice.ai/api/v1
======================================================================

Found 18 image(s)

──────────────────────────────────────────────────────────────────────
[1/18] DSCF1234.JPG
──────────────────────────────────────────────────────────────────────

📋 EXISTING METADATA
  Format: JPEG | Size: 3840x2160
  [META] Make: Apple
  [META] Model: iPhone SE (3rd generation)
  [META] DateTime: 2024:12:22 16:52:59

🖼️  RESIZING
  3840x2160 → 800x450

🤖 EXTRACTING KEYWORDS
    💰 Balance: $15.2347
  ✅ Keywords: car tire, snow chains, winter, snow...
     (12 total)

💾 SAVING
  Text file: DSCF1234_keywords.txt
  Enriched image: DSCF1234_enriched.JPG
  ✓ Keywords embedded in EXIF
  🔒 Original preserved
```

### Keywords File

```text
Source: DSCF1234.JPG
Model: google-gemma-3-27b-it
==================================================
KEYWORDS:
car tire, snow chains, winter, snow, vehicle, transportation, 
safety, ice, Dezent, rim, yellow, traction
==================================================
RAW AI RESPONSE:
car tire, snow chains, winter, snow, vehicle, transportation, 
safety, ice, Dezent, rim, yellow, traction, automotive, cold weather
```

## Troubleshooting

### "Model not found" (404)

Verify the model name in your `env.txt`. Check available models:

```bash
curl https://api.venice.ai/api/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### "Image validation failed" (400)

- Ensure model supports vision (check `VISION_KEYWORDS` in script)
- Try `google-gemma-3-27b-it` or `llama-3.2-90b-vision-instruct`

### Keywords not appearing in Windows Explorer

Windows caches thumbnail metadata. Try:
1. Right-click folder → Properties → General → Clear "Hidden files" checkbox
2. Or use a tool like ExifTool to verify: `exiftool -XPKeywords image.jpg`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- [Venice.ai](https://venice.ai) - Uncensored AI API
- [Pillow](https://python-pillow.org) - Python imaging library
- [piexif](https://github.com/hMatoba/Piexif) - EXIF manipulation

## Changelog

### v1.0.0 (2026)
- Initial release
- Vision-based keyword extraction
- EXIF metadata embedding
- Venice.ai best practices implementation
- Rate limiting and balance monitoring
