# ImageTagger — AI-powered image keyword tagger
# Author: Jussi Savola <jsavola@iki.fi>

import os
import re
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

# Ensure UTF-8 output on Windows where the default terminal encoding (cp1252)
# cannot represent emoji characters used in status messages.
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Configuration
MAX_DIMENSIONS = (800, 600)
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png')
DEFAULT_TAG = "jms"
DEFAULT_ENV_FILE = "env.txt"

# API provider constants
VENICE_BASE_URL = "https://api.venice.ai/api/v1"

# Adaptive inter-request throttle delay (seconds). Increases 10% on rate limit,
# decreases 2% on success, floored at 0.1s.
_throttle_delay = 0.1
VISION_KEYWORDS = ['vision', 'vl', 'llava', 'gemma', 'qwen2.5-vl', 'dolphin-vision', 'llava-1.6', 'qwen-vl', 'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo']

class APIConfig:
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

    def apply_overrides(self, model_override=None):
        if model_override:
            self.model = model_override
        self.is_vision = any(v in self.model.lower() for v in VISION_KEYWORDS)
        self.is_venice = 'venice.ai' in self.base_url

    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

def decode_piexif_value(raw_value):
    """
    Helper to decode values from piexif.
    Piexif returns XP fields as tuples of integers (bytes).
    """
    if raw_value is None:
        return ""
    
    # Case 1: It's a tuple of integers (common in piexif for XP fields)
    if isinstance(raw_value, tuple):
        try:
            # Convert tuple of ints to bytes, then decode UTF-16LE
            return bytes(raw_value).decode('utf-16le', errors='ignore').strip()
        except Exception:
            # Fallback if tuple contains non-integers or fails
            return str(raw_value)
    
    # Case 2: It's already bytes
    elif isinstance(raw_value, bytes):
        try:
            # Try UTF-16LE first (standard for XP fields)
            # Check for BOM or typical UTF-16 patterns
            if len(raw_value) >= 2 and (raw_value[1] == 0 or raw_value[0] == 0):
                return raw_value.decode('utf-16le', errors='ignore').strip()
            # Fallback to UTF-8 or ASCII
            return raw_value.decode('utf-8', errors='ignore').strip()
        except Exception:
            return str(raw_value)
    
    # Case 3: It's already a string or something else
    return str(raw_value).strip()

def check_already_processed(image_path, marker, use_xpcomment=False, verbose=False):
    """
    Check if image has already been processed by looking for the marker.
    """
    try:
        import piexif
        
        with Image.open(image_path) as img:
            if 'exif' not in img.info:
                if verbose:
                    print(f"      [DEBUG] No EXIF data found")
                return False
            
            exif_dict = piexif.load(img.info['exif'])
            
            target_value = None
            
            if use_xpcomment:
                # Check XPComment field
                raw_val = exif_dict.get("0th", {}).get(piexif.ImageIFD.XPComment)
                if raw_val:
                    target_value = decode_piexif_value(raw_val)
                    if verbose:
                        print(f"      [DEBUG] XPComment decoded: '{target_value}'")
                else:
                    if verbose:
                        print(f"      [DEBUG] No XPComment field found")
                    return False
            else:
                # Check XPKeywords field
                raw_val = exif_dict.get("0th", {}).get(piexif.ImageIFD.XPKeywords)
                if raw_val:
                    target_value = decode_piexif_value(raw_val)
                    if verbose:
                        print(f"      [DEBUG] XPKeywords decoded: '{target_value}'")
                else:
                    if verbose:
                        print(f"      [DEBUG] No XPKeywords field found")
                    return False

            if verbose:
                print(f"      [DEBUG] Looking for marker: '{marker}'")

            if target_value:
                # Split by comma for keywords list mode
                if not use_xpcomment:
                    keywords_list = [k.strip().lower() for k in target_value.split(',')]
                    if verbose:
                        print(f"      [DEBUG] Keywords list: {keywords_list}")
                    return marker.lower() in keywords_list
                else:
                    # Direct substring search for XPComment
                    return marker.lower() in target_value.lower()
            
            return False
                
    except ImportError:
        if verbose:
            print("      [DEBUG] piexif not installed")
        return False
    except Exception as e:
        if verbose:
            print(f"      [DEBUG] Error checking EXIF: {e}")
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

def strip_thinking(content):
    """Remove <think>...</think> reasoning blocks from model output."""
    return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()


def extract_from_reasoning(reasoning):
    """
    Extract the keywords answer from a raw 'reasoning' field (Ollama thinking models).
    The reasoning field is a plain-text chain-of-thought essay.  We look for the
    last 'Keywords:' / 'Tags:' marker; if absent, fall back to the last paragraph,
    which is typically where the model lands on its final answer.
    """
    for marker in ('Keywords:', 'Tags:', 'keywords:', 'tags:'):
        idx = reasoning.rfind(marker)
        if idx != -1:
            return reasoning[idx:]
    # No explicit marker — take the last non-empty paragraph
    paragraphs = [p.strip() for p in reasoning.split('\n\n') if p.strip()]
    return paragraphs[-1] if paragraphs else reasoning


def parse_ratelimit_reset(value):
    """Parse an API reset duration string like '1m4.637s', '30s', '500ms' into seconds."""
    if not value:
        return None
    value = value.strip()
    # Pure milliseconds (e.g. '500ms', '1ms') — must check before minutes
    m = re.fullmatch(r'(\d+)ms', value)
    if m:
        return int(m.group(1)) / 1000.0
    total = 0.0
    m = re.match(r'(\d+)m', value)
    if m:
        total += int(m.group(1)) * 60
        value = value[m.end():]
    m = re.match(r'([\d.]+)s', value)
    if m:
        total += float(m.group(1))
    return total if total > 0 else None


def call_AI_vision_for_keywords(base64_image, metadata_context, config, verbose=False, temperature=0.2):
    global _throttle_delay
    url = f"{config.base_url}/chat/completions"
    b64_clean = base64_image.strip().replace('\n', '').replace('\r', '')
    
    system_prompt = """You are a computer vision tagger. Analyze images and return ONLY a comma-separated list of 10-15 relevant keywords/tags.
Rules:
- Return ONLY keywords separated by commas (e.g., "car tire, snow chains, winter, snow")
- NO sentences, NO descriptions
- Include: objects, scenes, weather, colors, materials, brands, activities"""

    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_clean}"}},
        {"type": "text", "text": f"Generate keywords. Metadata context: {metadata_context[:200]} /no_think"}
    ]

    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": 200,
        "temperature": temperature,
        "think": False   # Ollama: suppress chain-of-thought for thinking models (ignored by other providers)
    }

    if config.is_venice:
        payload["venice_parameters"] = {
            "include_venice_system_prompt": False,
            "strip_thinking_response": True,
            "disable_thinking": True,
        }
    
    if verbose:
        print(f"\n  [VERBOSE] API Request Payload:")
        print(f"    Model: {config.model}")
    
    max_retries = 10
    base_delay = 1
    max_delay = 16

    for attempt in range(max_retries):
        try:
            time.sleep(_throttle_delay)
            response = requests.post(url, headers=config.get_headers(), json=payload, timeout=60)

            if verbose:
                print(f"\n  [VERBOSE] API Response Headers:")
                for key, value in response.headers.items():
                    if key.lower().startswith('x-') or key.lower() == 'cf-ray':
                        print(f"    {key}: {value}")

            balance_usd = None
            raw_balance = response.headers.get('x-venice-balance-usd')
            if raw_balance:
                balance_usd = float(raw_balance)
                print(f"    💰 Balance: ${balance_usd:.4f}")

            deprecation = response.headers.get('x-venice-model-deprecation-warning')
            if deprecation:
                print(f"    ⚠️  DEPRECATION: {deprecation}")

            if response.status_code == 200:
                _throttle_delay = max(0.1, _throttle_delay * 0.98)
                result = response.json()
                msg = result['choices'][0]['message']
                content = msg.get('content') or ''
                # Some thinking models (e.g. Ollama gemma4:e2b) return empty
                # content with the actual response in the 'reasoning' field.
                if not content.strip():
                    reasoning = msg.get('reasoning') or ''
                    content = extract_from_reasoning(reasoning) if reasoning else ''
                content = strip_thinking(content)

                if verbose:
                    print(f"\n  [VERBOSE] Raw AI Response:\n  {content}")

                keywords = parse_keywords(content)
                keywords = sanitize_keywords(keywords)
                return content, keywords, balance_usd

            elif response.status_code == 429 or response.status_code == 503:
                _throttle_delay *= 1.1
                # Prefer the reset time the API tells us to wait
                delay = None
                for remaining_key, reset_key in [
                    ('x-ratelimit-remaining-tokens', 'x-ratelimit-reset-tokens'),
                    ('x-ratelimit-remaining-requests', 'x-ratelimit-reset-requests'),
                ]:
                    remaining = response.headers.get(remaining_key)
                    if remaining is not None and int(remaining) == 0:
                        delay = parse_ratelimit_reset(response.headers.get(reset_key))
                        if delay is not None:
                            break
                if delay is None:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                print(f"    ⏳ Rate limited. Waiting {delay:.1f}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
                continue

            elif response.status_code == 401:
                return "ERROR_401: Invalid API key", [], balance_usd
            elif response.status_code == 404:
                return f"ERROR_404: Model '{config.model}' not found", [], balance_usd
            elif response.status_code == 400:
                error_data = response.json() if response.text else {}
                msg = error_data.get('error', {}).get('message', response.text[:200])
                return f"ERROR_400: {msg}", [], balance_usd
            else:
                cf_ray = response.headers.get('CF-RAY', 'N/A')
                return f"ERROR_{response.status_code}: CF-RAY={cf_ray}", [], balance_usd

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                print(f"    ⏳ Timeout. Retrying in {delay}s...")
                time.sleep(delay)
                continue
            return "ERROR: Request timeout after retries", [], None
        except Exception as e:
            return f"EXCEPTION: {str(e)}", [], None

    return "ERROR: Max retries exceeded", [], None

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

def sanitize_keywords(raw_keywords, max_keywords=20):
    """
    Normalize and validate model-produced keywords.

    Returns:
        clean_keywords: list[str]
    """
    clean = []
    seen = set()

    for k in raw_keywords:
        if not k:
            continue

        # Trim whitespace and common list markers / numbering
        k = k.strip()
        k = re.sub(r'^\s*[-*•]+\s*', '', k)          # "- cat", "* dog"
        k = re.sub(r'^\s*\d+[.)]\s*', '', k)         # "1. cat", "2) dog"
        k = k.strip().strip('"\'')

        # Remove trailing sentence punctuation
        k = k.rstrip('.,;:!?').strip()

        # Normalize internal whitespace
        k = re.sub(r'\s+', ' ', k)

        if not k:
            continue

        low = k.lower()

        # Reject trivial filler words
        if low in {'and', 'the', 'a', 'an'}:
            continue

        # Reject obvious prose / refusal-ish fragments
        if len(k) > 40:
            continue
        if any(ch in k for ch in '\n\r\t'):
            continue
        if len(k.split()) > 4:
            continue
        if re.search(r'[.!?]', k):
            continue

        # Optional character allowlist; keeps it conservative
        if not re.fullmatch(r"[A-Za-z0-9À-ÖØ-öø-ÿ _'&/+()-]+", k):
            continue

        if low not in seen:
            seen.add(low)
            clean.append(k)

        if len(clean) >= max_keywords:
            break

    return clean


def save_with_new_metadata(original_path, output_path, keywords, ai_raw_response,
                          original_exif_bytes=None, verbose=False, marker=None, use_xpcomment=False,
                          drop_gps=False, drop_datetime=False):
    """Write keywords into EXIF/IPTC metadata using piexif."""
    try:
        import piexif
        from piexif import ImageIFD, ExifIFD

        with Image.open(original_path) as img:
            if original_exif_bytes:
                exif_dict = piexif.load(original_exif_bytes)
            else:
                exif_dict = {"0th": {}, "1st": {}, "Exif": {}, "GPS": {}, "thumbnail": None}

            if drop_gps:
                exif_dict["GPS"] = {}

            if drop_datetime:
                for tag in [ImageIFD.DateTime]:
                    exif_dict["0th"].pop(tag, None)
                for tag in [ExifIFD.DateTimeOriginal, ExifIFD.DateTimeDigitized,
                            37521, 37522]:  # SubsecTimeOriginal, SubsecTimeDigitized (no named constant in piexif)
                    exif_dict["Exif"].pop(tag, None)
            
            # Write AI keywords to XPKeywords (always)
            keywords_str = ", ".join(keywords) if keywords else "AI analyzed"
            description = ai_raw_response[:500] if len(ai_raw_response) > 500 else ai_raw_response
            
            # Write processing marker based on -x flag
            if use_xpcomment:
                # Write marker ONLY to XPComment
                comment_text = f"Processed by ImageTagger: {marker}" if marker else "Processed by ImageTagger"
                exif_dict["0th"][ImageIFD.XPComment] = comment_text.encode('utf-16le')
                # Write clean keywords to XPKeywords
                exif_dict["0th"][ImageIFD.XPKeywords] = keywords_str.encode('utf-16le')
                marker_location = "XPComment"
            else:
                # Write marker to XPKeywords (append to keywords list)
                all_keywords = list(keywords) if keywords else []
                if marker and marker not in all_keywords:
                    all_keywords.append(marker)
                    keywords_str = ", ".join(all_keywords)
                
                exif_dict["0th"][ImageIFD.XPKeywords] = keywords_str.encode('utf-16le')
                marker_location = "XPKeywords"
            
            # Always write to XPSubject as well (copy of keywords)
            exif_dict["0th"][ImageIFD.XPSubject] = keywords_str.encode('utf-16le')
            
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
                if drop_gps:
                    print(f"    GPS data: stripped")
                if drop_datetime:
                    print(f"    Datetime fields: stripped")
            
            exif_bytes = piexif.dump(exif_dict)
            
            ext = original_path.suffix.lower()
            if ext in ('.jpg', '.jpeg'):
                img.save(output_path, 'JPEG', quality=95, exif=exif_bytes)
            else:
                img.save(output_path, 'PNG', exif=exif_bytes)
                
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
                   env_file=None, marker=None, use_xpcomment=False,
                   temperature=0.2, drop_gps=False, drop_datetime=False,
                   model_override=None):
    input_path = Path(input_dir).resolve()
    
    if not input_path.is_dir():
        print(f"Error: Input directory '{input_dir}' does not exist or is not a directory.")
        return

    if overwrite:
        print(f"\n{'='*70}")
        print("⚠️  OVERWRITE MODE ENABLED - Original files will be modified")
        print(f"{'='*70}")
    
    print(f"\n{'='*70}")
    print("IMAGETAGGER - AI Image Keyword Extractor")
    print(f"{'='*70}")
    
    try:
        config = APIConfig(env_file_path=env_file)
        config.apply_overrides(model_override=model_override)
        print(f"Model:  {config.model}")
        print(f"Vision: {'Yes' if config.is_vision else 'No (text-only)'}")
        print(f"Input:  {input_path}")
        mode_desc = "Overwrite original files" if overwrite else "Create copies in enriched subfolder"
        print(f"Mode:   {mode_desc}")
        print(f"Marker: '{marker}'")
        print(f"Marker field: {'XPComment' if use_xpcomment else 'XPKeywords (in keyword list)'}")
        print(f"Temperature: {temperature}")
        privacy_flags = []
        if drop_gps:
            privacy_flags.append("GPS stripped")
        if drop_datetime:
            privacy_flags.append("datetime stripped")
        if privacy_flags:
            print(f"Privacy: {', '.join(privacy_flags)}")
        if force:
            print(f"Force:  Yes (reprocessing all)")
        print(f"{'='*70}\n")
    except Exception as e:
        print(f"Config Error: {e}")
        return
    
    # Clean up any orphaned temp files left by a previous crashed run
    for orphan in input_path.glob("*_temp.*"):
        if orphan.suffix.lower() in SUPPORTED_FORMATS:
            try:
                orphan.unlink()
                print(f"  🧹 Removed orphaned temp file: {orphan.name}")
            except OSError as e:
                print(f"  ⚠️  Could not remove orphaned temp file {orphan.name}: {e}")

    images = [f for f in input_path.iterdir()
              if f.is_file()
              and f.suffix.lower() in SUPPORTED_FORMATS
              and not f.stem.endswith('_temp')]

    if not images:
        print(f"No supported images found in {input_path}")
        return

    print(f"Found {len(images)} image(s)\n")
    
    skipped = 0
    processed = 0
    errors = 0
    error_log = []  # list of (filename, reason)
    times_preprocess = []
    times_ai = []
    times_metadata = []

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
            t0 = time.time()

            # 1. Metadata
            print("\n📋 EXISTING METADATA")
            meta = extract_metadata(img_file, verbose=verbose)
            for line in meta['display_lines']:
                print(f"  {line}")

            # 2. Resize
            print(f"\n🖼️  RESIZING")
            b64, orig_sz, new_sz = resize_for_api(img_file)
            print(f"  {orig_sz[0]}x{orig_sz[1]} → {new_sz[0]}x{new_sz[1]}")

            t1 = time.time()

# 3. Get Keywords
            print(f"\n🤖 EXTRACTING KEYWORDS")
            meta_text = ", ".join(meta['ai_context'])
            ai_response, keywords, balance = call_AI_vision_for_keywords(b64, meta_text, config, verbose=verbose, temperature=temperature)

            # CRITICAL: Check for API errors OR Content Refusals before touching files
            is_error = ai_response.startswith("ERROR") or ai_response.startswith("EXCEPTION")
            is_refusal = not keywords  # sanitize_keywords already strips prose; empty list = real refusal

            if "ERROR_401" in ai_response:
                print(f"  ❌ FATAL: {ai_response}")
                print(f"  🛑 ABORTING - Invalid API key, no point continuing")
                return

            if "ERROR_404" in ai_response:
                print(f"  ❌ FATAL: {ai_response}")
                print(f"  🛑 ABORTING - Model not found, no point continuing")
                return

            if is_error or is_refusal:
                reason = ai_response[:120] if is_error else "Model refusal (no keywords extracted)"
                print(f"  ❌ FAILED: {ai_response}")
                print(f"  ⏭️  SKIPPING - Metadata left untouched due to error/refusal")
                errors += 1
                error_log.append((img_file.name, reason))
                continue # Skip to next image immediately

            # API balance/credits checks
            if balance is not None:
                if balance < 0.1:
                    print(f"\n  💸 FATAL: API balance ${balance:.4f} is below $0.10 — insufficient funds to continue")
                    print(f"  🛑 ABORTING")
                    return
                elif balance < 0.5:
                    print(f"\n  ⚠️  WARNING: API balance low (${balance:.4f}). Slowing down — waiting 60s before next image")
                    time.sleep(60)
                elif balance < 1.0:
                    print(f"\n  ⚠️  WARNING: API balance low (${balance:.4f}). Consider topping up soon")
            
            print(f"  ✅ Keywords: {', '.join(keywords[:5])}...")
            if len(keywords) > 5:
                print(f"     ({len(keywords)} total)")

            t2 = time.time()

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
            
            keywords_with_model = list(keywords) + [config.model]
            success = save_with_new_metadata(
                img_file, target_to_write, keywords_with_model, ai_response,
                orig_exif, verbose=verbose, marker=marker, use_xpcomment=use_xpcomment,
                drop_gps=drop_gps, drop_datetime=drop_datetime
            )
            
            if success:
                if overwrite:
                    # Retry loop: Windows Defender/Search Indexer can briefly lock a
                    # freshly written file, causing shutil.move to fail with WinError 32.
                    for _attempt in range(6):
                        try:
                            shutil.move(str(target_to_write), str(final_output))
                            break
                        except PermissionError:
                            if _attempt == 5:
                                raise
                            time.sleep(0.5)
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
                error_log.append((img_file.name, "Metadata write failed"))

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

            
            # Safe path printing: use relative path if possible, otherwise absolute
            try:
                log_display_path = analysis_file.relative_to(Path.cwd())
            except ValueError:
                log_display_path = analysis_file
            
            print(f"  📝 Log: {log_display_path}")

            t3 = time.time()
            dt_pre = t1 - t0
            dt_ai  = t2 - t1
            dt_meta = t3 - t2
            dt_total = t3 - t0
            print(f"\n⏱️  TIMING")
            print(f"  Preprocessing:     {dt_pre:.2f}s")
            print(f"  AI processing:     {dt_ai:.2f}s")
            print(f"  Metadata addition: {dt_meta:.2f}s")
            print(f"  Total:             {dt_total:.2f}s")

            if success:
                times_preprocess.append(dt_pre)
                times_ai.append(dt_ai)
                times_metadata.append(dt_meta)

        except Exception as e:
            print(f"\n  ❌ ERROR: {e}")
            errors += 1
            error_log.append((img_file.name, str(e)[:120]))
            if verbose:
                import traceback
                traceback.print_exc()
            if overwrite and temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except PermissionError:
                    print(f"  ⚠️  Could not delete temp file (file locked): {temp_file.name}")
    
    # Summary
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")
    print(f"  Total:     {len(images)}")

    if error_log:
        print(f"\n  Failed files:")
        for fname, reason in error_log:
            print(f"    ❌ {fname}: {reason}")

    if times_preprocess:
        sum_pre   = sum(times_preprocess)
        sum_ai    = sum(times_ai)
        sum_meta  = sum(times_metadata)
        sum_total = sum_pre + sum_ai + sum_meta
        n = len(times_preprocess)
        print(f"\n  Timing — totals ({n} image{'s' if n != 1 else ''} processed):")
        print(f"    Preprocessing:     {sum_pre:.2f}s")
        print(f"    AI processing:     {sum_ai:.2f}s")
        print(f"    Metadata addition: {sum_meta:.2f}s")
        print(f"    Total:             {sum_total:.2f}s")
        print(f"\n  Timing — averages per image:")
        print(f"    Preprocessing:     {sum_pre/n:.2f}s")
        print(f"    AI processing:     {sum_ai/n:.2f}s")
        print(f"    Metadata addition: {sum_meta/n:.2f}s")
        print(f"    Total:             {sum_total/n:.2f}s")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        prog='imagetagger',
        description="Tag images with AI-generated keywords using vision models.",
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
    parser.add_argument(
        '-m', '--model',
        type=str,
        default=None,
        help='Override the model from env.txt (e.g. gpt-4o-mini, qwen3-5-9b)'
    )
    parser.add_argument(
        '-T', '--temperature',
        type=float,
        default=0.2,
        help='AI sampling temperature 0.0–1.0 (default: 0.2; lower = more consistent)'
    )
    parser.add_argument(
        '--dropgps',
        action='store_true',
        help='Strip GPS coordinates from output images'
    )
    parser.add_argument(
        '--dropdatetime',
        action='store_true',
        help='Strip datetime fields from output images'
    )

    args = parser.parse_args()

    process_images(
        input_dir=args.directory,
        overwrite=args.overwrite,
        verbose=args.verbose,
        force=args.force,
        env_file=args.env,
        marker=args.tag,
        use_xpcomment=args.xpcomment,
        temperature=args.temperature,
        drop_gps=args.dropgps,
        drop_datetime=args.dropdatetime,
        model_override=args.model,
    )

if __name__ == "__main__":
    main()
