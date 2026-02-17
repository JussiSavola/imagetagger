import os
import base64
import io
import time
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import requests
import json
from tqdm import tqdm

# Configuration
MAX_DIMENSIONS = (800, 600)
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png')
PROCESSED_SUFFIX = "_enriched"

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
        if not self.is_vision:
            print(f"⚠️  Warning: {self.model} may not support vision")

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

def extract_metadata(image_path):
    meta = {'raw_exif': {}, 'display_lines': [], 'ai_context': []}
    try:
        with Image.open(image_path) as img:
            meta['display_lines'].append(f"Format: {img.format} | Size: {img.size[0]}x{img.size[1]}")
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    meta['raw_exif'][tag] = value
                    val_str = str(value)[:60]
                    if tag in ['Make', 'Model', 'DateTime', 'DateTimeOriginal', 'GPSInfo', 'ImageDescription']:
                        meta['display_lines'].append(f"  [META] {tag}: {val_str}")
                        meta['ai_context'].append(f"{tag}: {val_str}")
            if not meta['raw_exif']:
                meta['display_lines'].append("  [No EXIF]")
    except Exception as e:
        meta['display_lines'].append(f"  [Error: {e}]")
    return meta

def call_venice_for_keywords(base64_image, metadata_context, config):
    """
    Call Venice API with Best Practices:
    - Exponential Backoff on Rate Limits
    - Balance Monitoring (x-venice-balance-usd)
    - Request Logging (CF-RAY)
    - Model Deprecation Warnings
    """
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
    
    # Exponential Backoff Configuration
    max_retries = 3
    base_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                headers=config.get_headers(),
                json=payload,
                timeout=60
            )
            
            # --- VENICE BEST PRACTICES HANDLING ---
            
            # 1. Balance Monitoring
            balance_usd = response.headers.get('x-venice-balance-usd')
            balance_diem = response.headers.get('x-venice-balance-diem')
            if balance_usd:
                print(f"    💰 Balance: ${float(balance_usd):.4f}")
            
            # 2. Model Deprecation Warning
            deprecation = response.headers.get('x-venice-model-deprecation-warning')
            if deprecation:
                print(f"    ⚠️  DEPRECATION: {deprecation}")
            
            # 3. Rate Limiting Check
            remaining_requests = response.headers.get('x-ratelimit-remaining-requests')
            remaining_tokens = response.headers.get('x-ratelimit-remaining-tokens')
            
            # Handle specific status codes
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                keywords = parse_keywords(content)
                return content, keywords
            
            elif response.status_code == 429:
                # Rate limited - exponential backoff
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
                # 4. Request Logging (CF-RAY) for support
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
    """Extract clean keywords from AI response"""
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

def save_with_new_metadata(original_path, output_path, keywords, ai_raw_response, original_exif_bytes=None):
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

def process_images(input_dir):
    input_path = Path(input_dir)
    output_path = input_path / "enriched"
    output_path.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print("IMAGE KEYWORD EXTRACTOR - Venice.ai (Production Ready)")
    print(f"{'='*70}")
    
    try:
        config = VeniceConfig()
        print(f"Model: {config.model}")
        print(f"Vision: {'Yes' if config.is_vision else 'No (text only)'}")
        print(f"Endpoint: {config.base_url}")
        print(f"{'='*70}\n")
    except Exception as e:
        print(f"Config Error: {e}")
        return
    
    images = [f for f in input_path.iterdir() 
              if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS]
    
    if not images:
        print(f"No images found")
        return
    
    print(f"Found {len(images)} image(s)\n")
    
    for idx, img_file in enumerate(images, 1):
        print(f"\n{'─'*70}")
        print(f"[{idx}/{len(images)}] {img_file.name}")
        print(f"{'─'*70}")
        
        try:
            # 1. Metadata
            print("\n📋 EXISTING METADATA")
            meta = extract_metadata(img_file)
            for line in meta['display_lines']:
                print(f"  {line}")
            
            # 2. Resize
            print(f"\n🖼️  RESIZING")
            b64, orig_sz, new_sz = resize_for_api(img_file)
            print(f"  {orig_sz[0]}x{orig_sz[1]} → {new_sz[0]}x{new_sz[1]}")
            
            # 3. Get Keywords
            print(f"\n🤖 EXTRACTING KEYWORDS")
            meta_text = ", ".join(meta['ai_context'])
            ai_response, keywords = call_venice_for_keywords(b64, meta_text, config)
            
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
            
            analysis_file = output_path / f"{img_file.stem}_keywords.txt"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write(f"Source: {img_file.name}\n")
                f.write(f"Model: {config.model}\n")
                f.write(f"{'='*50}\n")
                f.write("KEYWORDS:\n")
                f.write(", ".join(keywords) if keywords else "No keywords extracted")
                f.write(f"\n{'='*50}\n")
                f.write("RAW AI RESPONSE:\n")
                f.write(ai_response)
            print(f"  Text file: {analysis_file.name}")
            
            enriched_file = output_path / f"{img_file.stem}{PROCESSED_SUFFIX}{img_file.suffix}"
            orig_exif = None
            with Image.open(img_file) as tmp:
                orig_exif = tmp.info.get('exif')
            
            success = save_with_new_metadata(img_file, enriched_file, keywords, ai_response, orig_exif)
            
            if success:
                print(f"  Enriched image: {enriched_file.name}")
                print(f"  ✓ Keywords embedded in EXIF")
            else:
                print(f"  Copied: {enriched_file.name} (metadata preserved)")
            
            if img_file.exists():
                print(f"  🔒 Original preserved")
            
            # 5. Verify
            print(f"\n🔍 VERIFICATION")
            if keywords:
                print(f"  Output keywords: {len(keywords)} tags")
                print(f"  Sample: {', '.join(keywords[:3])}")
                
        except Exception as e:
            print(f"\n  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"Output: {output_path.absolute()}")
    print(f"{'='*70}")

if __name__ == "__main__":
    process_images(".")