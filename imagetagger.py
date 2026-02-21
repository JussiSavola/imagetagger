import os
import base64
import io
import time
import argparse
import shutil
import sys
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import requests
import json

# Configuration
MAX_DIMENSIONS = (800, 600)
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png')
DEFAULT_TAG = "jms"
DEFAULT_ENV_FILE = "env.txt"

# Venice.ai specific constants
VENICE_BASE_URL = "https://api.venice.ai/api/v1"
VISION_KEYWORDS = ['vision', 'vl', 'llava', 'gemma', 'qwen2.5-vl', 'dolphin-vision', 'llava-1.6', 'qwen-vl']

class VeniceConfig:
    def __init__(self, env_file_path=None):
        # Determine env file location
        if env_file_path:
            env_file = Path(env_file_path).resolve()
        else:
            # Default: same directory as script
            script_dir = Path(__file__).parent.absolute()
            env_file = script_dir / DEFAULT_ENV_FILE
        
        if not env_file.exists():
            raise FileNotFoundError(f"Config file not found: {env_file}\nCreate it with: api_key=YOUR_KEY")
        
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
            raise ValueError(f"api_key required in {env_file}")
        
        self.base_url = self.config.get('api_base', VENICE_BASE_URL).rstrip('/')
        self.model = self.config.get('model', 'google-gemma-3-27b-it')
        
        self.is_vision = any(v in self.model.lower() for v in VISION_KEYWORDS)

    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

def check_already_processed(image_path, marker, use_xpcomment=False, verbose=False):
    """
    Check if image has already been processed by looking for the marker
    either in XPComment (if use_xpcomment=True) or XPKeywords (if use_xpcomment=False).
    Returns True if marker found, False otherwise.
    """
    try:
        import piexif
        
        with Image.open(image_path) as img:
            if 'exif' not in img.info:
                if verbose:
                    print(f"      [DEBUG] No EXIF data found in {image_path.name}")
                return False
            
            exif_dict = piexif.load(img.info['exif'])
            
            if use_xpcomment:
                # Check XPComment field
                xp_comment_raw = exif_dict.get("0th", {}).get(piexif.ImageIFD.XPComment, None)
                if xp_comment_raw is None:
                    if verbose:
                        print(f"      [DEBUG] No XPComment field found")
                    return False
                
                try:
                    # XPComment can be tuple (from piexif) or bytes
                    if isinstance(xp_comment_raw, tuple):
                        # Convert tuple of integers to bytes
                        comment_bytes = bytes(xp_comment_raw)
                        comment_str = comment_bytes.decode('utf-16le', errors='ignore').strip()
                    elif isinstance(xp_comment_raw, bytes):
                        comment_str = xp_comment_raw.decode('utf-16le', errors='ignore').strip()
                    else:
                        comment_str = str(xp_comment_raw).strip()
                    
                    if verbose:
                        print(f"      [DEBUG] XPComment value: '{comment_str}'")
                        print(f"      [DEBUG] Looking for marker: '{marker}'")
                    
                    return marker.lower() in comment_str.lower()
                    
                except Exception as e:
                    if verbose:
                        print(f"      [DEBUG] Error decoding XPComment: {e}")
                        print(f"      [DEBUG] Raw XPComment type: {type(xp_comment_raw)}, value: {xp_comment_raw}")
                    return False
            else:
                # Check XPKeywords field
                xp_keywords_raw = exif_dict.get("0th", {}).get(piexif.ImageIFD.XPKeywords, None)
                if xp_keywords_raw is None:
                    if verbose:
                        print(f"      [DEBUG] No XPKeywords field found")
                    return False
                
                try:
                    # XPKeywords can be tuple (from piexif) or bytes
                    if isinstance(xp_keywords_raw, tuple):
                        # Convert tuple of integers to bytes
                        keywords_bytes = bytes(xp_keywords_raw)
                        keywords_str = keywords_bytes.decode('utf-16le', errors='ignore').strip()
                    elif isinstance(xp_keywords_raw, bytes):
                        keywords_str = xp_keywords_raw.decode('utf-16le', errors='ignore').strip()
                    else:
                        keywords_str = str(xp_keywords_raw).strip()
                    
                    if verbose:
                        print(f"      [DEBUG] XPKeywords value: '{keywords_str}'")
                        print(f"      [DEBUG] Looking for marker: '{marker}'")
                    
                    # Split by comma and clean each keyword
                    keywords_list = [k.strip().lower() for k in keywords_str.split(',')]
                    if verbose:
                        print(f"      [DEBUG] Keywords list: {keywords_list}")
                    return marker.lower() in keywords_list
                    
                except Exception as e:
                    if verbose:
                        print(f"      [DEBUG] Error decoding XPKeywords: {e}")
                        print(f"      [DEBUG] Raw XPKeywords type: {type(xp_keywords_raw)}, value: {xp_keywords_raw}")
                    return False
                
    except ImportError:
        if verbose:
            print("      [DEBUG] piexif not installed")
        return False
    except Exception as e:
        if verbose:
            print(f"      [DEBUG] Error checking EXIF: {e}")
            import traceback
            traceback.print_exc()
        return False
    
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
                meta['display_lines'].append(f"  Mode: {img.mode} | Bits: {getattr(img, 'bits', 'N/A')}")
                
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    meta['raw_exif'][tag] = value
                    val_str = str(value)[:60]
                    
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
            
            balance_usd = response.headers.get('x-venice-balance-usd')
            if balance_usd:
                print(f"    💰 Balance: ${float(balance_usd):.4f}")
            
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

def save_with_new_metadata(original_path, output_path, keywords, ai_raw_response, 
                          original_exif_bytes=None, verbose=False, marker=None, use_xpcomment=False):
    """Write keywords into EXIF/IPTC metadata using piexif."""
    try:
        import piexif
        from piexif import ImageIFD, ExifIFD
        
        with Image.open(original_path) as img:
            if original_exif_bytes:
                exif_dict = piexif.load(original_exif_bytes)
            else:
                exif_dict = {"0th": {}, "1st": {}, "Exif": {}, "GPS": {}, "thumbnail": None}
            
            # Write AI keywords to XPKeywords (always)
            keywords_str = ", ".join(keywords) if keywords else "AI analyzed"
            description = ai_raw_response[:500] if len(ai_raw_response) > 500 else ai_raw_response
            
            # Write AI keywords
            exif_dict["0th"][ImageIFD.XPKeywords] = keywords_str.encode('utf-16le')
            exif_dict["0th"][ImageIFD.XPSubject] = keywords_str.encode('utf-16le')
            
            # Write processing marker based on -x flag
            if use_xpcomment:
                # Write marker ONLY to XPComment
                comment_text = f"Processed by ImageTagger: {marker}" if marker else "Processed by ImageTagger"
                exif_dict["0th"][ImageIFD.XPComment] = comment_text.encode('utf-16le')
                marker_location = "XPComment"
            else:
                # Write marker to XPKeywords (append to keywords list)
                all_keywords = list(keywords) if keywords else []
                if marker and marker not in all_keywords:
                    all_keywords.append(marker)
                    keywords_str = ", ".join(all_keywords)
                    exif_dict["0th"][ImageIFD.XPKeywords] = keywords_str.encode('utf-16le')
                    exif_dict["0th"][ImageIFD.XPSubject] = keywords_str.encode('utf-16le')
                marker_location = "XPKeywords"
            
            # Other standard fields
            exif_dict["0th"][ImageIFD.ImageDescription] = description.encode('utf-8')
            exif_dict["Exif"][ExifIFD.UserComment] = description.encode('utf-8')
            
            if verbose:
                print(f"\n  [VERBOSE] Writing EXIF fields:")
                print(f"    XPKeywords: {keywords_str}")
                if use_xpcomment:
                    print(f"    XPComment: {comment_text}")
                print(f"    ImageDescription: {description[:50]}...")
                print(f"    Processing marker '{marker}' in: {marker_location}")
            
            exif_bytes = piexif.dump(exif_dict)
            
            ext = original_path.suffix.lower()
            if ext in ('.jpg', '.jpeg'):
                img.save(output_path, 'JPEG', quality=95, exif=exif_bytes)
            else:
                img.save(output_path, 'PNG')
                
        return True
        
    except ImportError:
        print("      ⚠️  Install piexif to write metadata: pip install piexif")
        with Image.open(original_path) as img:
            save_kwargs = {}
            if original_exif_bytes:
                save_kwargs['exif'] = original_exif_bytes
            img.save(output_path, **save_kwargs)
        return False
    except Exception as e:
        print(f"      ⚠️  Metadata write failed: {e}")
        return False

def process_images(input_dir, overwrite=False, verbose=False, force=False, 
                   env_file=None, marker=None, use_xpcomment=False):
    input_path = Path(input_dir).resolve()
    
    if not input_path.is_dir():
        print(f"Error: Input directory '{input_dir}' does not exist or is not a directory.")
        return

    if overwrite:
        print(f"\n{'='*70}")
        print("⚠️  OVERWRITE MODE ENABLED - Original files will be modified")
        print(f"{'='*70}")
    
    print(f"\n{'='*70}")
    print("IMAGETAGGER - Venice.ai Image Keyword Extractor")
    print(f"{'='*70}")
    
    try:
        config = VeniceConfig(env_file_path=env_file)
        print(f"Model:  {config.model}")
        print(f"Vision: {'Yes' if config.is_vision else 'No (text-only)'}")
        print(f"Input:  {input_path}")
        mode_desc = "Overwrite original files" if overwrite else "Create copies in enriched subfolder"
        print(f"Mode:   {mode_desc}")
        print(f"Marker: '{marker}'")
        print(f"Marker field: {'XPComment' if use_xpcomment else 'XPKeywords (in keyword list)'}")
        if force:
            print(f"Force:  Yes (reprocessing all)")
        print(f"{'='*70}\n")
    except Exception as e:
        print(f"Config Error: {e}")
        return
    
    images = [f for f in input_path.iterdir() 
              if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS]
    
    if not images:
        print(f"No supported images found in {input_path}")
        return
    
    print(f"Found {len(images)} image(s)\n")
    
    skipped = 0
    processed = 0
    errors = 0
    
    for idx, img_file in enumerate(images, 1):
        print(f"\n{'─'*70}")
        print(f"[{idx}/{len(images)}] {img_file.name}")
        print(f"{'─'*70}")
        
        # Check if already processed (unless --force is used)
        if not force:
            already_done = check_already_processed(img_file, marker, use_xpcomment, verbose)
            if already_done:
                print(f"  ⏭️  SKIPPED - Already processed ('{marker}' marker found)")
                skipped += 1
                continue
            
            # Also check enriched copy if not in overwrite mode
            if not overwrite:
                enriched_check = input_path / "enriched" / f"{img_file.stem}_enriched{img_file.suffix}"
                if enriched_check.exists():
                    already_enriched = check_already_processed(enriched_check, marker, use_xpcomment, verbose)
                    if already_enriched:
                        print(f"  ⏭️  SKIPPED - Enriched copy already processed")
                        skipped += 1
                        continue
        
        temp_file = None
        
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
            ai_response, keywords = call_venice_for_keywords(b64, meta_text, config, verbose=verbose)
            
            if keywords:
                print(f"  ✅ Keywords: {', '.join(keywords[:5])}...")
                if len(keywords) > 5:
                    print(f"     ({len(keywords)} total)")
            else:
                print(f"  ⚠️  No keywords parsed")
                if ai_response.startswith("ERROR"):
                    print(f"     {ai_response}")
                    errors += 1
            
            # 4. Save
            print(f"\n💾 SAVING")
            
            if overwrite:
                temp_file = img_file.parent / f"{img_file.stem}_temp{img_file.suffix}"
                final_output = img_file
            else:
                output_dir = input_path / "enriched"
                output_dir.mkdir(exist_ok=True)
                temp_file = None
                final_output = output_dir / f"{img_file.stem}_enriched{img_file.suffix}"
            
            orig_exif = None
            with Image.open(img_file) as tmp:
                if 'exif' in tmp.info:
                    orig_exif = tmp.info['exif']

            target_to_write = temp_file if temp_file else final_output
            
            success = save_with_new_metadata(
                img_file, target_to_write, keywords, ai_response, 
                orig_exif, verbose=verbose, marker=marker, use_xpcomment=use_xpcomment
            )
            
            if success:
                if overwrite:
                    shutil.move(str(target_to_write), str(final_output))
                    print(f"  ✅ Overwritten: {final_output.name}")
                else:
                    print(f"  ✅ Enriched: {final_output.name}")
                field_name = "XPComment" if use_xpcomment else "XPKeywords"
                print(f"  ✅ Keywords embedded in EXIF (marker '{marker}' in {field_name})")
                processed += 1
            else:
                if overwrite and target_to_write.exists():
                    target_to_write.unlink()
                print(f"  ⚠️  Metadata preserved (no keywords written)")
                errors += 1

            # 5. Save text log
            log_dir = input_path / "enriched"
            log_dir.mkdir(exist_ok=True)

            analysis_file = log_dir / f"{img_file.stem}_keywords.txt"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write(f"Source: {img_file.name}\n")
                f.write(f"Model: {config.model}\n")
                f.write(f"Marker: {marker} ({'XPComment' if use_xpcomment else 'XPKeywords'})\n")
                f.write(f"{'='*50}\n")
                f.write("KEYWORDS:\n")
                f.write(", ".join(keywords) if keywords else "No keywords extracted")
                f.write(f"\n{'='*50}\n")
                f.write("RAW AI RESPONSE:\n")
                f.write(ai_response)
            print(f"  📝 Log: {analysis_file.relative_to(Path.cwd())}")
            
        except Exception as e:
            print(f"\n  ❌ ERROR: {e}")
            errors += 1
            if verbose:
                import traceback
                traceback.print_exc()
            if overwrite and temp_file and temp_file.exists():
                temp_file.unlink()
    
    # Summary
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")
    print(f"  Total:     {len(images)}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        prog='imagetagger',
        description="Tag images with AI-generated keywords using Venice.ai vision models.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python imagetagger.py                          # Process current directory
  python imagetagger.py -d ./photos             # Process specific directory  
  python imagetagger.py -e C:\\\\config\\\\env.txt   # Use specific env file
  python imagetagger.py -t "processed"          # Use custom marker tag
  python imagetagger.py -x                      # Store marker ONLY in XPComment field
  python imagetagger.py -f                      # Force reprocess all images
        """
    )
    parser.add_argument(
        '-d', '--directory',
        type=str,
        default='.',
        help='Directory containing images to process (default: current directory)'
    )
    parser.add_argument(
        '-e', '--env',
        type=str,
        default=None,
        help=f'Path to environment file with API key (default: script directory/{DEFAULT_ENV_FILE})'
    )
    parser.add_argument(
        '-o', '--overwrite',
        action='store_true',
        help='Overwrite original images instead of creating copies'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output (API details, all EXIF tags, etc.)'
    )
    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Force reprocessing of all images, even if already processed'
    )
    parser.add_argument(
        '-t', '--tag',
        type=str,
        default=DEFAULT_TAG,
        help=f'Custom marker tag for processed images (default: {DEFAULT_TAG})'
    )
    parser.add_argument(
        '-x', '--xpcomment',
        action='store_true',
        help='Store marker ONLY in XPComment field (not in keywords list)'
    )
    
    args = parser.parse_args()

    process_images(
        input_dir=args.directory,
        overwrite=args.overwrite,
        verbose=args.verbose,
        force=args.force,
        env_file=args.env,
        marker=args.tag,
        use_xpcomment=args.xpcomment
    )

if __name__ == "__main__":
    main()