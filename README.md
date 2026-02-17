

### imagetagger.py

```python
import os
import base64
import io
import time
import argparse
import shutil
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import requests
import json
from tqdm import tqdm

# Configuration
MAX_DIMENSIONS = (800, 600)
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png')

# Venice.ai specific constants
VENICE_BASE_URL = "https://api.venice.ai/api/v1"
VISION_KEYWORDS = ['vision', 'vl', 'llava', 'gemma', 'qwen2.5-vl', 'dolphin-vision', 'llava-1.6', 'qwen-vl']

class VeniceConfig:
    def __init__(self):
        script_dir = Path(__file__).parent.absolute()
        env_file = script_dir / "env.txt"
        
        if not env_file.exists():
            raise FileNotFoundError(f"Create env.txt in {script_dir} with: api_key=YOUR_KEY")
        
        self.config = {}
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    self.config[key.strip()] = value.strip()
                elif 'api_key' not in self.config:
                    self.config['api_key'] = line
        
        self.api_key = self.config.get('api_key')
        if not self.api_key:
            raise ValueError("api_key required in env.txt")
        
        self.base_url = self.config.get('api_base', VENICE_BASE_URL).rstrip('/')
        self.model = self.config.get('model', 'google-gemma-3-27b-it')
        
        self.is_vision = any(v in self.model.lower() for v in VISION_KEYWORDS)

    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

def resize_for_api(image_path):
    """Resize and encode to base64"""
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        orig_size = img.size
        img.thumbnail(MAX_DIMENSIONS, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85, optimize=True)
        b64 = base64.b64encode(buffer.getvalue()).decode('ascii')
        return b64, orig_size, img.size

def extract_metadata(image_path, verbose=False):
    meta = {'raw_exif': {}, 'display_lines': [], 'ai_context': []}
    try:
        with Image.open(image_path) as img:
            meta['display_lines'].append(f"Format: {img.format} | Size: {img.size[0]}x{img.size[1]}")
            if verbose:
                meta['display_lines'].append(f"  Mode: {img.mode} | Bits: {img.bits}")
                
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    meta['raw_exif'][tag] = value
                    val_str = str(value)[:60]
                    
                    # In verbose mode, show all tags. Otherwise just important ones.
                    if verbose:
                        meta['display_lines'].append(f"  [EXIF] {tag}: {val_str}")
                    elif tag in ['Make', 'Model', 'DateTime', 'DateTimeOriginal', 'GPSInfo', 'ImageDescription']:
                        meta['display_lines'].append(f"  [META] {tag}: {val_str}")
                        
                    if tag in ['Make', 'Model', 'DateTime', 'DateTimeOriginal', 'GPSInfo', 'ImageDescription']:
                        meta['ai_context'].append(f"{tag}: {val_str}")
                        
            if not meta['raw_exif']:
                meta['display_lines'].append("  [No EXIF]")
    except Exception as e:
        meta['display_lines'].append(f"  [Error: {e}]")
    return meta

def call_venice_for_keywords(base64_image, metadata_context, config, verbose=False):
    url = f"{config.base_url}/chat/completions"
    b64_clean = base64_image.strip().replace('\n', '').replace('\r', '')
    
    system_prompt = """You are a computer vision tagger. Analyze images and return ONLY a comma-separated list of 10-15 relevant keywords/tags.
Rules:
- Return ONLY keywords separated by commas (e.g., "car tire, snow chains, winter, snow")
- NO sentences, NO descriptions
- Include: objects, scenes, weather, colors, materials, brands, activities"""

    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_clean}"}},
        {"type": "text", "text": f"Generate keywords. Metadata context: {metadata_context[:200]}"}
    ]
    
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": 200,
        "temperature": 0.2
    }
    
    if verbose:
        print(f"\n  [VERBOSE] API Request Payload:")
        print(f"    Model: {config.model}")
        print(f"    Message Count: {len(payload['messages'])}")
    
    max_retries = 3
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=config.get_headers(), json=payload, timeout=60)
            
            if verbose:
                print(f"\n  [VERBOSE] API Response Headers:")
                for key, value in response.headers.items():
                    if key.lower().startswith('x-') or key.lower() == 'cf-ray':
                        print(f"    {key}: {value}")
            
            # Balance Monitoring
            balance_usd = response.headers.get('x-venice-balance-usd')
            if balance_usd:
                print(f"    💰 Balance: ${float(balance_usd):.4f}")
            
            # Deprecation Warning
            deprecation = response.headers.get('x-venice-model-deprecation-warning')
            if deprecation:
                print(f"    ⚠️  DEPRECATION: {deprecation}")
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                if verbose:
                    print(f"\n  [VERBOSE] Raw AI Response:\n  {content}")
                    
                keywords = parse_keywords(content)
                return content, keywords
            
            elif response.status_code == 429:
                delay = base_delay * (2 ** attempt)
                print(f"    ⏳ Rate limited. Waiting {delay}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
                continue
            
            elif response.status_code == 401:
                return "ERROR_401: Invalid API key", []
            elif response.status_code == 404:
                return f"ERROR_404: Model '{config.model}' not found", []
            elif response.status_code == 400:
                error_data = response.json() if response.text else {}
                msg = error_data.get('error', {}).get('message', response.text[:200])
                return f"ERROR_400: {msg}", []
            else:
                cf_ray = response.headers.get('CF-RAY', 'N/A')
                return f"ERROR_{response.status_code}: CF-RAY={cf_ray}", []
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"    ⏳ Timeout. Retrying in {delay}s...")
                time.sleep(delay)
                continue
            return "ERROR: Request timeout after retries", []
        except Exception as e:
            return f"EXCEPTION: {str(e)}", []
    
    return "ERROR: Max retries exceeded", []

def parse_keywords(ai_response):
    cleaned = ai_response.replace("Keywords:", "").replace("Tags:", "")
    cleaned = cleaned.replace("Here are the keywords:", "").replace("Here is a list of keywords:", "")
    
    if ',' in cleaned:
        keywords = [k.strip() for k in cleaned.split(',')]
    else:
        keywords = [k.strip() for k in cleaned.split('\n') if k.strip()]
    
    clean_list = []
    for k in keywords:
        k = k.strip().rstrip('.').strip('"\'')
        if k.lower() not in ['', 'and', 'the', 'a', 'an']:
            clean_list.append(k)
    
    return clean_list[:20]

def save_with_new_metadata(original_path, output_path, keywords, ai_raw_response, original_exif_bytes=None, verbose=False):
    """Write keywords into EXIF/IPTC metadata using piexif"""
    try:
        import piexif
        from piexif import ImageIFD, ExifIFD
        
        with Image.open(original_path) as img:
            if original_exif_bytes:
                exif_dict = piexif.load(original_exif_bytes)
            else:
                exif_dict = {"0th": {}, "1st": {}, "Exif": {}, "GPS": {}, "thumbnail": None}
            
            keywords_str = ", ".join(keywords) if keywords else "AI analyzed"
            description = ai_raw_response[:500] if len(ai_raw_response) > 500 else ai_raw_response
            
            if keywords:
                exif_dict["0th"][ImageIFD.XPKeywords] = keywords_str.encode('utf-16le')
                exif_dict["0th"][ImageIFD.XPSubject] = keywords_str.encode('utf-16le')
            
            exif_dict["0th"][ImageIFD.ImageDescription] = description.encode('utf-8')
            exif_dict["Exif"][ExifIFD.UserComment] = description.encode('utf-8')
            
            if verbose:
                print(f"\n  [VERBOSE] Writing EXIF fields:")
                print(f"    XPKeywords: {keywords_str}")
                print(f"    ImageDescription: {description[:50]}...")
            
            exif_bytes = piexif.dump(exif_dict)
            
            ext = original_path.suffix.lower()
            if ext in ('.jpg', '.jpeg'):
                img.save(output_path, 'JPEG', quality=95, exif=exif_bytes)
            else:
                img.save(output_path, 'PNG')
                
        return True
        
    except ImportError:
        print("      ⚠️  Install piexif: pip install piexif")
        with Image.open(original_path) as img:
            save_kwargs = {}
            if original_exif_bytes:
                save_kwargs['exif'] = original_exif_bytes
            img.save(output_path, **save_kwargs)
        return False
    except Exception as e:
        print(f"      ⚠️  Metadata write failed: {e}")
        return False

def process_images(input_dir, overwrite=False, verbose=False):
    input_path = Path(input_dir).resolve()
    
    # Determine output strategy
    if overwrite:
        output_path = input_path  # Same directory
        print(f"\n{'='*70}")
        print("⚠️  OVERWRITE MODE ENABLED - Original files will be modified")
        print(f"{'='*70}")
    else:
        output_path = input_path / "enriched"
        output_path.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print("IMAGE KEYWORD EXTRACTOR - Venice.ai")
    print(f"{'='*70}")
    
    try:
        config = VeniceConfig()
        print(f"Model: {config.model}")
        print(f"Vision: {'Yes' if config.is_vision else 'No (text only)'}")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(f"{'='*70}\n")
    except Exception as e:
        print(f"Config Error: {e}")
        return
    
    images = [f for f in input_path.iterdir() 
              if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS]
    
    if not images:
        print(f"No images found in {input_path}")
        return
    
    print(f"Found {len(images)} image(s)\n")
    
    for idx, img_file in enumerate(images, 1):
        print(f"\n{'─'*70}")
        print(f"[{idx}/{len(images)}] {img_file.name}")
        print(f"{'─'*70}")
        
        # Define output file path
        if overwrite:
            # Use a temp file for safety during overwrite
            temp_output = img_file.parent / f"{img_file.stem}_temp{img_file.suffix}"
            final_output = img_file
        else:
            temp_output = None
            final_output = output_path / f"{img_file.stem}_enriched{img_file.suffix}"
        
        try:
            # 1. Metadata
            print("\n📋 EXISTING METADATA")
            meta = extract_metadata(img_file, verbose=verbose)
            for line in meta['display_lines']:
                print(f"  {line}")
            
            # 2. Resize
            print(f"\n🖼️  RESIZING")
            b64, orig_sz, new_sz = resize_for_api(img_file)
            print(f"  {orig_sz[0]}x{orig_sz[1]} → {new_sz[0]}x{new_sz[1]}")
            
            # 3. Get Keywords
            print(f"\n🤖 EXTRACTING KEYWORDS")
            meta_text = ", ".join(meta['ai_context'])
            
            # Determine output path for saving logic
            save_target = temp_output if temp_output else final_output
            
            ai_response, keywords = call_venice_for_keywords(b64, meta_text, config, verbose=verbose)
            
            if keywords:
                print(f"  ✅ Keywords: {', '.join(keywords[:5])}...")
                if len(keywords) > 5:
                    print(f"     ({len(keywords)} total)")
            else:
                print(f"  ⚠️  No keywords parsed")
                if ai_response.startswith("ERROR"):
                    print(f"     {ai_response}")
            
            # 4. Save
            print(f"\n💾 SAVING")
            
            # Get original EXIF
            orig_exif = None
            with Image.open(img_file) as tmp:
                orig_exif = tmp.info.get('exif')
            
            success = save_with_new_metadata(img_file, save_target, keywords, ai_response, orig_exif, verbose=verbose)
            
            # Handle overwrite atomic operation
            if overwrite and success:
                # Replace original with temp file
                shutil.move(str(save_target), str(final_output))
                print(f"  Overwritten: {final_output.name}")
                print(f"  ✓ Keywords embedded in EXIF")
            elif success:
                print(f"  Enriched image: {final_output.name}")
                print(f"  ✓ Keywords embedded in EXIF")
            else:
                if overwrite and save_target.exists():
                    save_target.unlink()  # Clean up temp file on failure
                print(f"  ⚠️  Metadata preserved (no keywords written)")
            
            # Save text log (always in same folder as image or 'enriched' folder)
            if overwrite:
                log_dir = input_path
            else:
                log_dir = output_path
                
            analysis_file = log_dir / f"{img_file.stem}_keywords.txt"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write(f"Source: {img_file.name}\n")
                f.write(f"Model: {config.model}\n")
                f.write(f"{'='*50}\n")
                f.write("KEYWORDS:\n")
                f.write(", ".join(keywords) if keywords else "No keywords extracted")
                f.write(f"\n{'='*50}\n")
                f.write("RAW AI RESPONSE:\n")
                f.write(ai_response)
                
        except Exception as e:
            print(f"\n  ❌ ERROR: {e}")
            import traceback
            if verbose:
                traceback.print_exc()
            # Clean up temp file if it exists
            if overwrite and temp_output and temp_output.exists():
                temp_output.unlink()
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")

def main():
    parser = argparse.ArgumentParser(
        description="Enrich image metadata using Venice.ai vision models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python imagetagger.py                         # Process current dir, save to ./enriched
  python imagetagger.py -d ./photos             # Process ./photos
  python imagetagger.py -d ./photos -o          # Overwrite originals in ./photos
  python imagetagger.py -v                      # Verbose output
        """
    )
    
    parser.add_argument(
        '-d', '--directory',
        type=str,
        default='.',
        help='Directory containing images (default: current directory)'
    )
    
    parser.add_argument(
        '-o', '--overwrite',
        action='store_true',
        help='Overwrite original images instead of creating enriched copies'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed debug information'
    )
    
    args = parser.parse_args()
    
    process_images(
        input_dir=args.directory,
        overwrite=args.overwrite,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
```

---

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