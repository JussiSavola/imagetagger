#!/usr/bin/env python3
"""
BatchImageTagger — resumable OpenAI Batch API image keyword tagger

Based on imagetagger.py by Jussi Savola, adapted for asynchronous OpenAI Batch
processing with local on-disk state.

Behavior summary:
- First run in a directory with no batch state:
  * scans images
  * skips already-processed images unless --force
  * prepares a chunk of images under --max-batch-bytes
  * uploads requests.jsonl and creates an OpenAI batch job
- Later runs:
  * poll batch status
  * download output/error files when available
  * apply EXIF/log updates for completed results
  * once current chunk is resolved, automatically prepare the next chunk

State lives under .batch_imagetagger/
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from PIL import Image
from PIL.ExifTags import TAGS


# Ensure UTF-8 output on Windows where the default terminal encoding (cp1252)
# cannot represent emoji characters used in status messages.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_DIMENSIONS = (800, 600)
SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png")
DEFAULT_TAG = "jms"
DEFAULT_ENV_FILE = "env.txt"
OPENAI_BASE_URL = "https://api.openai.com/v1"
STATE_DIRNAME = ".batch_imagetagger"
REQUESTS_JSONL = "requests.jsonl"
OUTPUT_JSONL = "output.jsonl"
ERROR_OUTPUT_JSONL = "error_output.jsonl"
MANIFEST_JSON = "manifest.json"
BATCH_META_JSON = "batch_meta.json"
APPLY_JOURNAL_JSONL = "apply_journal.jsonl"
ARCHIVE_DIRNAME = "history"
DEFAULT_MAX_BATCH_BYTES = 150_000_000  # keep healthy margin below 200 MB limit
MAX_BATCH_BYTES_HARD = 190_000_000     # refuse to approach the 200 MB hard limit

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_mtime_utc(path: Path) -> str:
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_atomic_temp_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    ts = int(time.time() * 1000)
    return path.with_name(f"{path.name}.tmp.{pid}.{ts}")


def _replace_with_retries(src: Path, dst: Path, retries: int = 8, delay: float = 0.25) -> None:
    last_exc = None
    for attempt in range(retries):
        try:
            os.replace(str(src), str(dst))
            return
        except PermissionError as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
                continue
            raise
        except OSError as e:
            last_exc = e
            # Fallback for some SMB/Windows cases: remove then replace.
            if dst.exists():
                try:
                    dst.unlink()
                    os.replace(str(src), str(dst))
                    return
                except Exception:
                    pass
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
                continue
            raise
    if last_exc:
        raise last_exc


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = make_atomic_temp_path(path)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        _replace_with_retries(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def parse_jsonl(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def bool_from_str(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class APIConfig:
    def __init__(self, env_file_path: Optional[str] = None):
        if env_file_path:
            env_file = Path(env_file_path).resolve()
        else:
            script_dir = Path(__file__).parent.absolute()
            env_file = script_dir / DEFAULT_ENV_FILE

        if not env_file.exists():
            raise FileNotFoundError(
                f"Config file not found: {env_file}\n"
                "Create it with at least: api_key=YOUR_KEY"
            )

        self.config: Dict[str, str] = {}
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    self.config[key.strip()] = value.strip()
                elif "api_key" not in self.config:
                    self.config["api_key"] = line

        self.api_key = self.config.get("api_key")
        if not self.api_key:
            raise ValueError(f"api_key required in {env_file}")

        self.base_url = self.config.get("api_base", OPENAI_BASE_URL).rstrip("/")
        self.model = self.config.get("model", "gpt-4.1-mini")
        self.organization = self.config.get("organization")
        self.project = self.config.get("project")
        self.prompt_cache_key = self.config.get("prompt_cache_key")
        self.request_timeout = float(self.config.get("request_timeout", "120"))

    def apply_overrides(self, model_override: Optional[str] = None) -> None:
        if model_override:
            self.model = model_override

    def get_headers(self, *, json_content: bool = True) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        if json_content:
            headers["Content-Type"] = "application/json"
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        if self.project:
            headers["OpenAI-Project"] = self.project
        return headers


# ---------------------------------------------------------------------------
# Existing imagetagger helpers, lightly adapted
# ---------------------------------------------------------------------------


def decode_piexif_value(raw_value):
    if raw_value is None:
        return ""
    if isinstance(raw_value, tuple):
        try:
            return bytes(raw_value).decode("utf-16le", errors="ignore").strip()
        except Exception:
            return str(raw_value)
    elif isinstance(raw_value, bytes):
        try:
            if len(raw_value) >= 2 and (raw_value[1] == 0 or raw_value[0] == 0):
                return raw_value.decode("utf-16le", errors="ignore").strip()
            return raw_value.decode("utf-8", errors="ignore").strip()
        except Exception:
            return str(raw_value)
    return str(raw_value).strip()


def check_already_processed(image_path: Path, marker: str, use_xpcomment: bool = False, verbose: bool = False) -> bool:
    try:
        import piexif

        with Image.open(image_path) as img:
            if "exif" not in img.info:
                if verbose:
                    print("      [DEBUG] No EXIF data found")
                return False

            exif_dict = piexif.load(img.info["exif"])
            target_value = None

            if use_xpcomment:
                raw_val = exif_dict.get("0th", {}).get(piexif.ImageIFD.XPComment)
                if raw_val:
                    target_value = decode_piexif_value(raw_val)
                    if verbose:
                        print(f"      [DEBUG] XPComment decoded: '{target_value}'")
                else:
                    if verbose:
                        print("      [DEBUG] No XPComment field found")
                    return False
            else:
                raw_val = exif_dict.get("0th", {}).get(piexif.ImageIFD.XPKeywords)
                if raw_val:
                    target_value = decode_piexif_value(raw_val)
                    if verbose:
                        print(f"      [DEBUG] XPKeywords decoded: '{target_value}'")
                else:
                    if verbose:
                        print("      [DEBUG] No XPKeywords field found")
                    return False

            if verbose:
                print(f"      [DEBUG] Looking for marker: '{marker}'")

            if target_value:
                if not use_xpcomment:
                    keywords_list = [k.strip().lower() for k in target_value.split(",")]
                    return marker.lower() in keywords_list
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


def resize_for_api(image_path: Path) -> Tuple[str, Tuple[int, int], Tuple[int, int]]:
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        orig_size = img.size
        img.thumbnail(MAX_DIMENSIONS, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85, optimize=True)
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        return b64, orig_size, img.size


def extract_metadata(image_path: Path, verbose: bool = False) -> dict:
    meta = {"raw_exif": {}, "display_lines": [], "ai_context": []}
    try:
        with Image.open(image_path) as img:
            meta["display_lines"].append(f"Format: {img.format} | Size: {img.size[0]}x{img.size[1]}")
            if verbose:
                meta["display_lines"].append(f"  Mode: {img.mode} | Bits: {getattr(img, 'bits', 'N/A')}")

            if hasattr(img, "_getexif") and img._getexif():
                exif = img._getexif()
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    meta["raw_exif"][tag] = value
                    val_str = str(value)[:60]
                    if verbose:
                        meta["display_lines"].append(f"  [EXIF] {tag}: {val_str}")
                    elif tag in ["Make", "Model", "DateTime", "DateTimeOriginal", "GPSInfo", "ImageDescription"]:
                        meta["display_lines"].append(f"  [META] {tag}: {val_str}")

                    if tag in ["Make", "Model", "DateTime", "DateTimeOriginal", "GPSInfo", "ImageDescription"]:
                        meta["ai_context"].append(f"{tag}: {val_str}")

            if not meta["raw_exif"]:
                meta["display_lines"].append("  [No EXIF]")
    except Exception as e:
        meta["display_lines"].append(f"  [Error: {e}]")
    return meta


def parse_keywords(ai_response: str) -> List[str]:
    cleaned = ai_response.replace("Keywords:", "").replace("Tags:", "")
    cleaned = cleaned.replace("Here are the keywords:", "").replace("Here is a list of keywords:", "")
    if "," in cleaned:
        keywords = [k.strip() for k in cleaned.split(",")]
    else:
        keywords = [k.strip() for k in cleaned.split("\n") if k.strip()]

    clean_list = []
    for k in keywords:
        k = k.strip().rstrip(".").strip('"\'')
        if k.lower() not in ["", "and", "the", "a", "an"]:
            clean_list.append(k)
    return clean_list[:20]


def sanitize_keywords(raw_keywords: List[str], max_keywords: int = 20) -> List[str]:
    clean: List[str] = []
    seen = set()
    for k in raw_keywords:
        if not k:
            continue
        k = k.strip()
        k = re.sub(r"^\s*[-*•]+\s*", "", k)
        k = re.sub(r"^\s*\d+[.)]\s*", "", k)
        k = k.strip().strip('"\'')
        k = k.rstrip(".,;:!?").strip()
        k = re.sub(r"\s+", " ", k)
        if not k:
            continue
        low = k.lower()
        if low in {"and", "the", "a", "an"}:
            continue
        if len(k) > 40 or len(k.split()) > 4:
            continue
        if re.search(r"[.!?]", k):
            continue
        if not re.fullmatch(r"[A-Za-z0-9À-ÖØ-öø-ÿ _'&/+()\-]+", k):
            continue
        if low not in seen:
            seen.add(low)
            clean.append(k)
        if len(clean) >= max_keywords:
            break
    return clean


def save_with_new_metadata(
    original_path: Path,
    output_path: Path,
    keywords: List[str],
    ai_raw_response: str,
    original_exif_bytes: Optional[bytes] = None,
    verbose: bool = False,
    marker: Optional[str] = None,
    use_xpcomment: bool = False,
    drop_gps: bool = False,
    drop_datetime: bool = False,
) -> bool:
    try:
        import piexif
        from piexif import ExifIFD, ImageIFD

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

            keywords_str = ", ".join(keywords) if keywords else "AI analyzed"
            description = ai_raw_response[:500] if len(ai_raw_response) > 500 else ai_raw_response

            if use_xpcomment:
                comment_text = f"Processed by BatchImageTagger: {marker}" if marker else "Processed by BatchImageTagger"
                exif_dict["0th"][ImageIFD.XPComment] = comment_text.encode("utf-16le")
                exif_dict["0th"][ImageIFD.XPKeywords] = keywords_str.encode("utf-16le")
                marker_location = "XPComment"
            else:
                all_keywords = list(keywords) if keywords else []
                if marker and marker not in all_keywords:
                    all_keywords.append(marker)
                    keywords_str = ", ".join(all_keywords)
                exif_dict["0th"][ImageIFD.XPKeywords] = keywords_str.encode("utf-16le")
                marker_location = "XPKeywords"

            exif_dict["0th"][ImageIFD.XPSubject] = keywords_str.encode("utf-16le")
            exif_dict["0th"][ImageIFD.ImageDescription] = description.encode("utf-8", errors="ignore")
            exif_dict["Exif"][ExifIFD.UserComment] = description.encode("utf-8", errors="ignore")

            if verbose:
                print("\n  [VERBOSE] Writing EXIF fields:")
                print(f"    XPKeywords: {keywords_str}")
                if use_xpcomment:
                    print(f"    XPComment: {comment_text}")
                print(f"    Processing marker '{marker}' in: {marker_location}")
                if drop_gps:
                    print(f"    GPS data: stripped")
                if drop_datetime:
                    print(f"    Datetime fields: stripped")

            exif_bytes = piexif.dump(exif_dict)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            ext = original_path.suffix.lower()
            if ext in (".jpg", ".jpeg"):
                img.save(output_path, "JPEG", quality=95, exif=exif_bytes)
            else:
                img.save(output_path, "PNG", exif=exif_bytes)
        return True
    except ImportError:
        print("      ⚠️  Install piexif to write metadata: pip install piexif")
        with Image.open(original_path) as img:
            save_kwargs = {}
            if original_exif_bytes:
                save_kwargs["exif"] = original_exif_bytes
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, **save_kwargs)
        return False
    except Exception as e:
        print(f"      ⚠️  Metadata write failed: {e}")
        return False


# ---------------------------------------------------------------------------
# State model
# ---------------------------------------------------------------------------

class JobState(str, Enum):
    NEW = "new"
    PREPARING = "preparing"
    SUBMITTED = "submitted"
    WAITING = "waiting"
    OUTPUT_READY = "output_ready"
    APPLYING = "applying"
    PARTIALLY_APPLIED = "partially_applied"
    COMPLETED = "completed"
    COMPLETED_WITH_FAILURES = "completed_with_failures"
    ERROR = "error"


class ItemStatus(str, Enum):
    PENDING_LOCAL = "pending_local"
    PREPARED = "prepared"
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    APPLIED = "applied"
    SKIPPED = "skipped"
    FAILED_REMOTE = "failed_remote"
    FAILED_PARSE = "failed_parse"
    FAILED_APPLY = "failed_apply"
    INVALIDATED = "invalidated"


@dataclass
class BatchSettings:
    model: str
    marker: str
    use_xpcomment: bool
    overwrite: bool
    force: bool
    max_dimensions: List[int] = field(default_factory=lambda: [MAX_DIMENSIONS[0], MAX_DIMENSIONS[1]])
    supported_formats: List[str] = field(default_factory=lambda: list(SUPPORTED_FORMATS))
    max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES
    temperature: float = 0.2
    drop_gps: bool = False
    drop_datetime: bool = False


@dataclass
class RequestInfo:
    image_format: str = ""
    original_dimensions: List[int] = field(default_factory=list)
    resized_dimensions: List[int] = field(default_factory=list)
    metadata_context: str = ""
    prompt_cache_key: Optional[str] = None
    estimated_line_bytes: int = 0


@dataclass
class ProcessingInfo:
    status: str = ItemStatus.PENDING_LOCAL.value
    status_reason: str = ""
    prepared_at: Optional[str] = None
    submitted_at: Optional[str] = None
    completed_at: Optional[str] = None
    applied_at: Optional[str] = None
    failed_at: Optional[str] = None
    batch_id: Optional[str] = None
    attempt_count: int = 0
    skip_reason: Optional[str] = None
    error_reason: Optional[str] = None


@dataclass
class ResultInfo:
    raw_response_text: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    keyword_count: int = 0
    usage: Optional[Dict[str, Any]] = None


@dataclass
class LocalIntegrity:
    source_exists: bool = True
    source_changed_since_prepare: bool = False
    output_written: bool = False
    log_written: bool = False


@dataclass
class ManifestItem:
    custom_id: str
    source_path: str
    source_relpath: str
    source_size_bytes: int
    source_mtime_utc: str
    output_path: str
    output_relpath: str
    log_path: str
    log_relpath: str
    request: RequestInfo = field(default_factory=RequestInfo)
    processing: ProcessingInfo = field(default_factory=ProcessingInfo)
    result: ResultInfo = field(default_factory=ResultInfo)
    local_integrity: LocalIntegrity = field(default_factory=LocalIntegrity)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ManifestItem":
        return cls(
            custom_id=data["custom_id"],
            source_path=data["source_path"],
            source_relpath=data["source_relpath"],
            source_size_bytes=data["source_size_bytes"],
            source_mtime_utc=data["source_mtime_utc"],
            output_path=data["output_path"],
            output_relpath=data["output_relpath"],
            log_path=data["log_path"],
            log_relpath=data["log_relpath"],
            request=RequestInfo(**data.get("request", {})),
            processing=ProcessingInfo(**data.get("processing", {})),
            result=ResultInfo(**data.get("result", {})),
            local_integrity=LocalIntegrity(**data.get("local_integrity", {})),
        )


@dataclass
class Manifest:
    schema_version: int = 1
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    directory: str = ""
    settings_snapshot: Optional[BatchSettings] = None
    items: List[ManifestItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "directory": self.directory,
            "settings_snapshot": asdict(self.settings_snapshot) if self.settings_snapshot else None,
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        return cls(
            schema_version=data.get("schema_version", 1),
            created_at=data.get("created_at", utc_now_iso()),
            updated_at=data.get("updated_at", utc_now_iso()),
            directory=data.get("directory", ""),
            settings_snapshot=BatchSettings(**data["settings_snapshot"]) if data.get("settings_snapshot") else None,
            items=[ManifestItem.from_dict(x) for x in data.get("items", [])],
        )

    def item_map(self) -> Dict[str, ManifestItem]:
        return {item.custom_id: item for item in self.items}


@dataclass
class CounterSummary:
    total_items: int = 0
    pending_local: int = 0
    prepared: int = 0
    submitted: int = 0
    completed: int = 0
    applied: int = 0
    skipped: int = 0
    failed_remote: int = 0
    failed_parse: int = 0
    failed_apply: int = 0
    invalidated: int = 0


@dataclass
class BatchFlags:
    requests_uploaded: bool = False
    batch_created: bool = False
    output_downloaded: bool = False
    error_output_downloaded: bool = False
    apply_started: bool = False
    apply_finished: bool = False


@dataclass
class BatchMeta:
    schema_version: int = 1
    tool_name: str = "batch_imagetagger"
    tool_version: str = "0.1"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    directory: str = ""
    settings: Optional[BatchSettings] = None
    job_state: str = JobState.NEW.value
    job_state_reason: str = ""
    active_batch_id: Optional[str] = None
    active_input_file_id: Optional[str] = None
    active_output_file_id: Optional[str] = None
    active_error_file_id: Optional[str] = None
    remote_status: Optional[str] = None
    current_chunk_index: int = 0
    current_chunk_bytes: int = 0
    counters: CounterSummary = field(default_factory=CounterSummary)
    flags: BatchFlags = field(default_factory=BatchFlags)
    last_poll_time: Optional[str] = None
    last_poll_note: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "directory": self.directory,
            "settings": asdict(self.settings) if self.settings else None,
            "job_state": self.job_state,
            "job_state_reason": self.job_state_reason,
            "active_batch_id": self.active_batch_id,
            "active_input_file_id": self.active_input_file_id,
            "active_output_file_id": self.active_output_file_id,
            "active_error_file_id": self.active_error_file_id,
            "remote_status": self.remote_status,
            "current_chunk_index": self.current_chunk_index,
            "current_chunk_bytes": self.current_chunk_bytes,
            "counters": asdict(self.counters),
            "flags": asdict(self.flags),
            "last_poll_time": self.last_poll_time,
            "last_poll_note": self.last_poll_note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BatchMeta":
        return cls(
            schema_version=data.get("schema_version", 1),
            tool_name=data.get("tool_name", "batch_imagetagger"),
            tool_version=data.get("tool_version", "0.1"),
            created_at=data.get("created_at", utc_now_iso()),
            updated_at=data.get("updated_at", utc_now_iso()),
            directory=data.get("directory", ""),
            settings=BatchSettings(**data["settings"]) if data.get("settings") else None,
            job_state=data.get("job_state", JobState.NEW.value),
            job_state_reason=data.get("job_state_reason", ""),
            active_batch_id=data.get("active_batch_id"),
            active_input_file_id=data.get("active_input_file_id"),
            active_output_file_id=data.get("active_output_file_id"),
            active_error_file_id=data.get("active_error_file_id"),
            remote_status=data.get("remote_status"),
            current_chunk_index=data.get("current_chunk_index", 0),
            current_chunk_bytes=data.get("current_chunk_bytes", 0),
            counters=CounterSummary(**data.get("counters", {})),
            flags=BatchFlags(**data.get("flags", {})),
            last_poll_time=data.get("last_poll_time"),
            last_poll_note=data.get("last_poll_note"),
        )


# ---------------------------------------------------------------------------
# State disk I/O
# ---------------------------------------------------------------------------


def state_dir(root_dir: Path) -> Path:
    return root_dir / STATE_DIRNAME


def state_file(root_dir: Path, name: str) -> Path:
    return state_dir(root_dir) / name


def save_manifest(root_dir: Path, manifest: Manifest) -> None:
    manifest.updated_at = utc_now_iso()
    write_json_atomic(state_file(root_dir, MANIFEST_JSON), manifest.to_dict())


def load_manifest(root_dir: Path) -> Manifest:
    return Manifest.from_dict(read_json(state_file(root_dir, MANIFEST_JSON)))


def save_meta(root_dir: Path, meta: BatchMeta) -> None:
    meta.updated_at = utc_now_iso()
    write_json_atomic(state_file(root_dir, BATCH_META_JSON), meta.to_dict())


def load_meta(root_dir: Path) -> BatchMeta:
    return BatchMeta.from_dict(read_json(state_file(root_dir, BATCH_META_JSON)))


def append_journal(root_dir: Path, record: dict) -> None:
    append_jsonl(state_file(root_dir, APPLY_JOURNAL_JSONL), record)


# ---------------------------------------------------------------------------
# Manifest / item helpers
# ---------------------------------------------------------------------------


def make_custom_id(index: int) -> str:
    return f"img-{index:06d}"


def build_output_paths(root_dir: Path, image_path: Path, overwrite: bool) -> Tuple[Path, Path]:
    enriched_dir = root_dir / "enriched"
    if overwrite:
        output_path = image_path
        log_path = enriched_dir / f"{image_path.stem}_keywords.txt"
    else:
        output_path = enriched_dir / f"{image_path.stem}_enriched{image_path.suffix}"
        log_path = enriched_dir / f"{image_path.stem}_keywords.txt"
    return output_path, log_path


def build_manifest(root_dir: Path, settings: BatchSettings, *, verbose: bool = False) -> Manifest:
    # Clean up orphaned temp files left by a previous crashed run
    for orphan in root_dir.glob("*_temp.*"):
        if orphan.suffix.lower() in SUPPORTED_FORMATS:
            try:
                orphan.unlink()
                print(f"  🧹 Removed orphaned temp file: {orphan.name}")
            except OSError as e:
                print(f"  ⚠️  Could not remove orphaned temp file {orphan.name}: {e}")

    images = [
        p for p in sorted(root_dir.iterdir(), key=lambda x: x.name.lower())
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_FORMATS
        and not p.stem.endswith("_temp")
    ]

    print(f"📂 Scanning {len(images)} image(s) for already-processed markers...")

    items: List[ManifestItem] = []
    skipped_count = 0
    for idx, image_path in enumerate(images, start=1):
        output_path, log_path = build_output_paths(root_dir, image_path, settings.overwrite)

        skipped = False
        skip_reason = None
        if not settings.force:
            if check_already_processed(image_path, settings.marker, settings.use_xpcomment, verbose=verbose):
                skipped = True
                skip_reason = f"Marker '{settings.marker}' already present in source"
            elif (not settings.overwrite) and output_path.exists() and check_already_processed(output_path, settings.marker, settings.use_xpcomment, verbose=verbose):
                skipped = True
                skip_reason = "Enriched copy already processed"

        if skipped:
            skipped_count += 1

        item = ManifestItem(
            custom_id=make_custom_id(idx),
            source_path=str(image_path),
            source_relpath=str(image_path.relative_to(root_dir)),
            source_size_bytes=image_path.stat().st_size,
            source_mtime_utc=file_mtime_utc(image_path),
            output_path=str(output_path),
            output_relpath=str(output_path.relative_to(root_dir)),
            log_path=str(log_path),
            log_relpath=str(log_path.relative_to(root_dir)),
            processing=ProcessingInfo(
                status=ItemStatus.SKIPPED.value if skipped else ItemStatus.PENDING_LOCAL.value,
                status_reason=skip_reason or "Pending selection for future batch",
                skip_reason=skip_reason,
            ),
        )
        items.append(item)

    pending = len(images) - skipped_count
    print(f"  ✅ {pending} pending, {skipped_count} already processed (skipped)")
    return Manifest(directory=str(root_dir), settings_snapshot=settings, items=items)


def refresh_item_source_integrity(item: ManifestItem) -> None:
    src = Path(item.source_path)
    item.local_integrity.source_exists = src.exists()
    if not src.exists():
        item.local_integrity.source_changed_since_prepare = True
        return
    current_size = src.stat().st_size
    current_mtime = file_mtime_utc(src)
    changed = current_size != item.source_size_bytes or current_mtime != item.source_mtime_utc
    item.local_integrity.source_changed_since_prepare = changed


def recompute_counters(manifest: Manifest) -> CounterSummary:
    c = CounterSummary(total_items=len(manifest.items))
    for item in manifest.items:
        status = item.processing.status
        if status == ItemStatus.PENDING_LOCAL.value:
            c.pending_local += 1
        elif status == ItemStatus.PREPARED.value:
            c.prepared += 1
        elif status == ItemStatus.SUBMITTED.value:
            c.submitted += 1
        elif status == ItemStatus.COMPLETED.value:
            c.completed += 1
        elif status == ItemStatus.APPLIED.value:
            c.applied += 1
        elif status == ItemStatus.SKIPPED.value:
            c.skipped += 1
        elif status == ItemStatus.FAILED_REMOTE.value:
            c.failed_remote += 1
        elif status == ItemStatus.FAILED_PARSE.value:
            c.failed_parse += 1
        elif status == ItemStatus.FAILED_APPLY.value:
            c.failed_apply += 1
        elif status == ItemStatus.INVALIDATED.value:
            c.invalidated += 1
    return c


def active_items(manifest: Manifest, batch_id: Optional[str]) -> List[ManifestItem]:
    if not batch_id:
        return []
    return [item for item in manifest.items if item.processing.batch_id == batch_id and item.processing.status not in {ItemStatus.APPLIED.value, ItemStatus.SKIPPED.value}]


def unresolved_active_items(manifest: Manifest, batch_id: Optional[str]) -> List[ManifestItem]:
    if not batch_id:
        return []
    unresolved = []
    for item in manifest.items:
        if item.processing.batch_id != batch_id:
            continue
        if item.processing.status in {
            ItemStatus.PREPARED.value,
            ItemStatus.SUBMITTED.value,
            ItemStatus.COMPLETED.value,
        }:
            unresolved.append(item)
    return unresolved


def any_pending_local(manifest: Manifest) -> bool:
    return any(item.processing.status == ItemStatus.PENDING_LOCAL.value for item in manifest.items)


def derive_job_state(meta: BatchMeta, manifest: Manifest) -> str:
    if meta.active_batch_id:
        unresolved = unresolved_active_items(manifest, meta.active_batch_id)
        if meta.remote_status in {"validating", "in_progress", "finalizing", "cancelling"}:
            return JobState.WAITING.value
        if any(item.processing.status == ItemStatus.COMPLETED.value for item in manifest.items if item.processing.batch_id == meta.active_batch_id):
            return JobState.OUTPUT_READY.value
        if unresolved:
            return JobState.WAITING.value
    if any(item.processing.status == ItemStatus.COMPLETED.value for item in manifest.items):
        return JobState.OUTPUT_READY.value
    if any_pending_local(manifest):
        return JobState.PREPARING.value
    failures = any(item.processing.status in {
        ItemStatus.FAILED_REMOTE.value,
        ItemStatus.FAILED_PARSE.value,
        ItemStatus.FAILED_APPLY.value,
        ItemStatus.INVALIDATED.value,
    } for item in manifest.items)
    if failures:
        return JobState.COMPLETED_WITH_FAILURES.value
    return JobState.COMPLETED.value


def sync_meta_from_manifest(meta: BatchMeta, manifest: Manifest) -> None:
    meta.counters = recompute_counters(manifest)
    meta.job_state = derive_job_state(meta, manifest)

    if meta.job_state == JobState.COMPLETED.value:
        meta.job_state_reason = "All selected images have been applied; no pending images remain"
        meta.flags.apply_finished = True
    elif meta.job_state == JobState.COMPLETED_WITH_FAILURES.value:
        meta.job_state_reason = "Processing complete; some images failed or were invalidated"
        meta.flags.apply_finished = True
    elif meta.job_state == JobState.OUTPUT_READY.value:
        meta.job_state_reason = "Downloaded results are ready to be applied"
    elif meta.job_state == JobState.WAITING.value:
        meta.job_state_reason = f"Waiting for batch output (remote status: {meta.remote_status})"
    elif meta.job_state == JobState.PREPARING.value:
        meta.job_state_reason = "Pending images available for the next batch chunk"


# ---------------------------------------------------------------------------
# OpenAI Batch API helpers (requests-based)
# ---------------------------------------------------------------------------


def build_openai_chat_body(
    *,
    model: str,
    base64_image: str,
    metadata_context: str,
    prompt_cache_key: Optional[str] = None,
    temperature: float = 0.2,
) -> dict:
    b64_clean = base64_image.strip().replace("\n", "").replace("\r", "")
    system_prompt = (
        "You are a computer vision tagger. Analyze images and return ONLY a "
        "comma-separated list of 10-15 relevant keywords/tags.\n"
        "Rules:\n"
        "- Return ONLY keywords separated by commas\n"
        "- NO sentences, NO descriptions\n"
        "- Include: objects, scenes, weather, colors, materials, brands, activities"
    )
    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_clean}"}},
        {"type": "text", "text": f"Generate keywords. Metadata context: {metadata_context[:200]}"},
    ]
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 200,
        "temperature": temperature,
    }
    if prompt_cache_key:
        body["prompt_cache_key"] = prompt_cache_key
    return body


def estimate_jsonl_line_bytes(custom_id: str, body: dict) -> int:
    line_obj = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }
    return len((json.dumps(line_obj, ensure_ascii=False) + "\n").encode("utf-8"))


def post_json(config: APIConfig, endpoint: str, payload: dict) -> dict:
    url = f"{config.base_url}{endpoint}"
    response = requests.post(url, headers=config.get_headers(json_content=True), json=payload, timeout=config.request_timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"POST {endpoint} failed: HTTP {response.status_code} {response.text[:500]}")
    return response.json()


def get_json(config: APIConfig, endpoint: str) -> dict:
    url = f"{config.base_url}{endpoint}"
    response = requests.get(url, headers=config.get_headers(json_content=False), timeout=config.request_timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"GET {endpoint} failed: HTTP {response.status_code} {response.text[:500]}")
    return response.json()


def upload_file(config: APIConfig, path: Path) -> str:
    url = f"{config.base_url}/files"
    headers = config.get_headers(json_content=False)
    with open(path, "rb") as f:
        files = {"file": (path.name, f, "application/jsonl")}
        data = {"purpose": "batch"}
        response = requests.post(url, headers=headers, files=files, data=data, timeout=config.request_timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"File upload failed: HTTP {response.status_code} {response.text[:500]}")
    return response.json()["id"]


def create_batch(config: APIConfig, input_file_id: str, *, metadata: Optional[dict] = None) -> dict:
    payload = {
        "input_file_id": input_file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
    }
    if metadata:
        payload["metadata"] = metadata
    return post_json(config, "/batches", payload)


def retrieve_batch(config: APIConfig, batch_id: str) -> dict:
    return get_json(config, f"/batches/{batch_id}")


def cancel_batch(config: APIConfig, batch_id: str) -> dict:
    return post_json(config, f"/batches/{batch_id}/cancel", {})


def download_file_content(config: APIConfig, file_id: str) -> bytes:
    url = f"{config.base_url}/files/{file_id}/content"
    response = requests.get(url, headers=config.get_headers(json_content=False), timeout=config.request_timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"File content download failed: HTTP {response.status_code} {response.text[:500]}")
    return response.content


# ---------------------------------------------------------------------------
# Batch chunk preparation
# ---------------------------------------------------------------------------


def prepare_next_chunk(root_dir: Path, meta: BatchMeta, manifest: Manifest, config: APIConfig, *, verbose: bool = False) -> int:
    requests_path = state_file(root_dir, REQUESTS_JSONL)
    if requests_path.exists():
        requests_path.unlink()

    total_bytes = 0
    selected = 0
    chunk_index = meta.current_chunk_index + 1

    pending_items = [item for item in manifest.items if item.processing.status == ItemStatus.PENDING_LOCAL.value]
    total_pending = len(pending_items)
    print(f"\n🔧 PREPARING CHUNK {chunk_index}  ({total_pending} pending image(s))")
    print("─" * 70)

    for item in pending_items:
        image_path = Path(item.source_path)
        if not image_path.exists():
            item.processing.status = ItemStatus.INVALIDATED.value
            item.processing.status_reason = "Source image missing before batch preparation"
            item.processing.error_reason = item.processing.status_reason
            print(f"  [{selected+1}/{total_pending}] ⚠️  Missing: {image_path.name}")
            continue

        print(f"  [{selected+1}/{total_pending}] 🖼️  {image_path.name}", end="", flush=True)
        meta_info = extract_metadata(image_path, verbose=False)
        metadata_context = ", ".join(meta_info.get("ai_context", []))
        b64_image, orig_sz, resized_sz = resize_for_api(image_path)
        body = build_openai_chat_body(
            model=meta.settings.model,
            base64_image=b64_image,
            metadata_context=metadata_context,
            prompt_cache_key=config.prompt_cache_key,
            temperature=meta.settings.temperature if meta.settings else 0.2,
        )
        line_bytes = estimate_jsonl_line_bytes(item.custom_id, body)

        if line_bytes > MAX_BATCH_BYTES_HARD:
            item.processing.status = ItemStatus.FAILED_PARSE.value
            item.processing.status_reason = "Single request too large for safe batch submission"
            item.processing.error_reason = f"Estimated line size {line_bytes} bytes exceeds hard limit"
            print(f"  ❌ too large ({line_bytes} bytes)")
            continue

        if selected > 0 and total_bytes + line_bytes > meta.settings.max_batch_bytes:
            print(f"  ⏸  batch cap reached, remaining deferred to next chunk")
            break
        if total_bytes + line_bytes > MAX_BATCH_BYTES_HARD:
            print(f"  ⏸  hard limit reached")
            break

        print(f"  ({orig_sz[0]}x{orig_sz[1]} → {resized_sz[0]}x{resized_sz[1]}, {line_bytes//1024}KB)")

        line_obj = {
            "custom_id": item.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        append_jsonl(requests_path, line_obj)

        item.request.image_format = image_path.suffix.lower()
        item.request.original_dimensions = [orig_sz[0], orig_sz[1]]
        item.request.resized_dimensions = [resized_sz[0], resized_sz[1]]
        item.request.metadata_context = metadata_context
        item.request.prompt_cache_key = config.prompt_cache_key
        item.request.estimated_line_bytes = line_bytes
        item.processing.status = ItemStatus.PREPARED.value
        item.processing.status_reason = f"Prepared for chunk {chunk_index}"
        item.processing.prepared_at = utc_now_iso()
        item.processing.attempt_count += 1
        total_bytes += line_bytes
        selected += 1

    print("─" * 70)
    print(f"  Prepared: {selected} image(s)  |  Estimated size: {total_bytes/1_000_000:.2f} MB")

    meta.current_chunk_index = chunk_index if selected else meta.current_chunk_index
    meta.current_chunk_bytes = total_bytes
    sync_meta_from_manifest(meta, manifest)
    save_manifest(root_dir, manifest)
    save_meta(root_dir, meta)

    return selected


def submit_prepared_chunk(root_dir: Path, meta: BatchMeta, manifest: Manifest, config: APIConfig) -> None:
    requests_path = state_file(root_dir, REQUESTS_JSONL)
    if not requests_path.exists():
        raise FileNotFoundError(f"Missing {requests_path}")

    size_mb = requests_path.stat().st_size / 1_000_000
    print(f"\n⬆️  Uploading requests.jsonl ({size_mb:.2f} MB)...", end="", flush=True)
    input_file_id = upload_file(config, requests_path)
    print(f"  ✅ file_id: {input_file_id}")

    print(f"🚀 Creating batch job...", end="", flush=True)
    batch = create_batch(
        config,
        input_file_id,
        metadata={
            "tool": "batch_imagetagger",
            "directory": meta.directory,
            "chunk_index": str(meta.current_chunk_index),
        },
    )

    batch_id = batch["id"]
    print(f"  ✅ Batch created: {batch_id}  (status: {batch.get('status')})")
    print(f"  ⏳ Processing will complete within 24h. Re-run this script to poll and apply results.")
    meta.active_input_file_id = input_file_id
    meta.active_batch_id = batch_id
    meta.active_output_file_id = batch.get("output_file_id")
    meta.active_error_file_id = batch.get("error_file_id")
    meta.remote_status = batch.get("status")
    meta.flags.requests_uploaded = True
    meta.flags.batch_created = True
    meta.flags.output_downloaded = False
    meta.flags.error_output_downloaded = False
    meta.flags.apply_started = False
    meta.flags.apply_finished = False
    meta.job_state = JobState.SUBMITTED.value
    meta.job_state_reason = f"Batch {batch_id} created successfully"

    for item in manifest.items:
        if item.processing.status == ItemStatus.PREPARED.value:
            item.processing.status = ItemStatus.SUBMITTED.value
            item.processing.status_reason = f"Submitted in batch {batch_id}"
            item.processing.submitted_at = utc_now_iso()
            item.processing.batch_id = batch_id

    sync_meta_from_manifest(meta, manifest)
    save_manifest(root_dir, manifest)
    save_meta(root_dir, meta)


# ---------------------------------------------------------------------------
# Poll / download / ingest
# ---------------------------------------------------------------------------


def poll_active_batch(root_dir: Path, meta: BatchMeta, config: APIConfig) -> dict:
    if not meta.active_batch_id:
        raise ValueError("No active batch id")
    batch = retrieve_batch(config, meta.active_batch_id)
    meta.remote_status = batch.get("status")
    meta.active_output_file_id = batch.get("output_file_id")
    meta.active_error_file_id = batch.get("error_file_id")
    meta.last_poll_time = utc_now_iso()
    counts = batch.get("request_counts") or {}
    completed = counts.get("completed")
    failed = counts.get("failed")
    total = counts.get("total")
    if completed is not None and failed is not None and total is not None:
        meta.last_poll_note = f"completed={completed}, failed={failed}, total={total}"
    else:
        meta.last_poll_note = None
    sync_meta_from_manifest(meta, load_manifest(root_dir)) if False else None
    save_meta(root_dir, meta)
    return batch


def download_active_outputs(root_dir: Path, meta: BatchMeta, config: APIConfig) -> None:
    if meta.active_output_file_id and not meta.flags.output_downloaded:
        print(f"⬇️  Downloading output results...", end="", flush=True)
        raw = download_file_content(config, meta.active_output_file_id)
        with open(state_file(root_dir, OUTPUT_JSONL), "wb") as f:
            f.write(raw)
        meta.flags.output_downloaded = True
        print(f"  ✅ {len(raw)/1_000:.1f} KB")
    if meta.active_error_file_id and not meta.flags.error_output_downloaded:
        print(f"⬇️  Downloading error file...", end="", flush=True)
        raw = download_file_content(config, meta.active_error_file_id)
        with open(state_file(root_dir, ERROR_OUTPUT_JSONL), "wb") as f:
            f.write(raw)
        meta.flags.error_output_downloaded = True
        print(f"  ✅ {len(raw)/1_000:.1f} KB")
    save_meta(root_dir, meta)


def ingest_output_jsonl(root_dir: Path, meta: BatchMeta, manifest: Manifest) -> Tuple[int, int]:
    output_path = state_file(root_dir, OUTPUT_JSONL)
    if not output_path.exists():
        return 0, 0
    item_by_id = manifest.item_map()
    completed = 0
    failed_parse = 0

    for record in parse_jsonl(output_path):
        custom_id = record.get("custom_id")
        if not custom_id or custom_id not in item_by_id:
            continue
        item = item_by_id[custom_id]
        if item.processing.batch_id != meta.active_batch_id:
            continue
        if item.processing.status in {ItemStatus.APPLIED.value, ItemStatus.FAILED_REMOTE.value, ItemStatus.FAILED_PARSE.value, ItemStatus.FAILED_APPLY.value, ItemStatus.INVALIDATED.value}:
            continue

        error_obj = record.get("error")
        response_obj = record.get("response")
        if error_obj:
            item.processing.status = ItemStatus.FAILED_REMOTE.value
            item.processing.status_reason = "Error object present in batch output"
            item.processing.failed_at = utc_now_iso()
            item.processing.error_reason = json.dumps(error_obj, ensure_ascii=False)
            continue
        if not response_obj:
            item.processing.status = ItemStatus.FAILED_PARSE.value
            item.processing.status_reason = "Missing response object in batch output line"
            item.processing.failed_at = utc_now_iso()
            item.processing.error_reason = item.processing.status_reason
            failed_parse += 1
            continue

        status_code = response_obj.get("status_code")
        body = response_obj.get("body", {})
        if status_code != 200:
            item.processing.status = ItemStatus.FAILED_REMOTE.value
            item.processing.status_reason = f"HTTP {status_code} in batch response"
            item.processing.failed_at = utc_now_iso()
            item.processing.error_reason = json.dumps(body, ensure_ascii=False)[:1000]
            continue
        try:
            content = body["choices"][0]["message"]["content"]
        except Exception as e:
            item.processing.status = ItemStatus.FAILED_PARSE.value
            item.processing.status_reason = f"Malformed response body: {e}"
            item.processing.failed_at = utc_now_iso()
            item.processing.error_reason = item.processing.status_reason
            failed_parse += 1
            continue

        keywords = sanitize_keywords(parse_keywords(content))
        if not keywords:
            item.processing.status = ItemStatus.FAILED_PARSE.value
            item.processing.status_reason = "No usable keywords after parsing/sanitizing"
            item.processing.failed_at = utc_now_iso()
            item.processing.error_reason = item.processing.status_reason
            item.result.raw_response_text = content
            failed_parse += 1
            continue

        item.processing.status = ItemStatus.COMPLETED.value
        item.processing.status_reason = "Successful result downloaded"
        item.processing.completed_at = utc_now_iso()
        item.processing.error_reason = None
        item.result.raw_response_text = content
        item.result.keywords = keywords
        item.result.keyword_count = len(keywords)
        item.result.usage = body.get("usage")
        completed += 1
    save_manifest(root_dir, manifest)
    return completed, failed_parse


def ingest_error_jsonl(root_dir: Path, meta: BatchMeta, manifest: Manifest) -> int:
    error_path = state_file(root_dir, ERROR_OUTPUT_JSONL)
    if not error_path.exists():
        return 0
    item_by_id = manifest.item_map()
    failed = 0
    for record in parse_jsonl(error_path):
        custom_id = record.get("custom_id")
        if not custom_id or custom_id not in item_by_id:
            continue
        item = item_by_id[custom_id]
        if item.processing.batch_id != meta.active_batch_id:
            continue
        if item.processing.status in {ItemStatus.APPLIED.value, ItemStatus.FAILED_REMOTE.value, ItemStatus.FAILED_PARSE.value, ItemStatus.FAILED_APPLY.value, ItemStatus.INVALIDATED.value}:
            continue
        error_obj = record.get("error")
        if error_obj:
            item.processing.status = ItemStatus.FAILED_REMOTE.value
            item.processing.status_reason = "Request listed in batch error file"
            item.processing.failed_at = utc_now_iso()
            item.processing.error_reason = json.dumps(error_obj, ensure_ascii=False)
            failed += 1
    save_manifest(root_dir, manifest)
    return failed


# ---------------------------------------------------------------------------
# Apply phase
# ---------------------------------------------------------------------------


def invalidate_if_source_changed(item: ManifestItem) -> bool:
    refresh_item_source_integrity(item)
    if not item.local_integrity.source_exists:
        item.processing.status = ItemStatus.INVALIDATED.value
        item.processing.status_reason = "Source image no longer exists"
        item.processing.error_reason = item.processing.status_reason
        return True
    if item.local_integrity.source_changed_since_prepare:
        item.processing.status = ItemStatus.INVALIDATED.value
        item.processing.status_reason = "Source image changed after batch submission"
        item.processing.error_reason = item.processing.status_reason
        return True
    return False


def apply_completed_item(root_dir: Path, meta: BatchMeta, item: ManifestItem, *, verbose: bool = False) -> bool:
    if invalidate_if_source_changed(item):
        append_journal(root_dir, {
            "time": utc_now_iso(),
            "custom_id": item.custom_id,
            "action": "invalidated",
            "reason": item.processing.error_reason,
        })
        return False

    src = Path(item.source_path)
    dst = Path(item.output_path)
    log_path = Path(item.log_path)
    append_journal(root_dir, {
        "time": utc_now_iso(),
        "custom_id": item.custom_id,
        "action": "apply_started",
        "source": item.source_relpath,
    })

    try:
        orig_exif = None
        with Image.open(src) as tmp:
            if "exif" in tmp.info:
                orig_exif = tmp.info["exif"]

        keywords_with_model = list(item.result.keywords)
        if meta.settings and meta.settings.model not in keywords_with_model:
            keywords_with_model.append(meta.settings.model)

        target_to_write = dst
        temp_target = make_atomic_temp_path(dst)
        target_to_write = temp_target

        ok = save_with_new_metadata(
            src,
            target_to_write,
            keywords_with_model,
            item.result.raw_response_text or "",
            original_exif_bytes=orig_exif,
            verbose=verbose,
            marker=meta.settings.marker if meta.settings else DEFAULT_TAG,
            use_xpcomment=meta.settings.use_xpcomment if meta.settings else False,
            drop_gps=meta.settings.drop_gps if meta.settings else False,
            drop_datetime=meta.settings.drop_datetime if meta.settings else False,
        )
        if not ok:
            raise RuntimeError("save_with_new_metadata() returned False")

        _replace_with_retries(temp_target, dst)

        log_path.parent.mkdir(parents=True, exist_ok=True)
        temp_log = make_atomic_temp_path(log_path)
        try:
            with open(temp_log, "w", encoding="utf-8") as f:
                f.write(f"Source: {src.name}\n")
                if meta.settings:
                    f.write(f"Model: {meta.settings.model}\n")
                f.write(
                    f"Marker: {meta.settings.marker if meta.settings else DEFAULT_TAG} "
                    f"({'XPComment' if meta.settings and meta.settings.use_xpcomment else 'XPKeywords'})\n"
                )
                f.write("=" * 50 + "\n")
                f.write("KEYWORDS:\n")
                f.write(", ".join(item.result.keywords) if item.result.keywords else "No keywords extracted")
                f.write("\n" + "=" * 50 + "\n")
                f.write("RAW AI RESPONSE:\n")
                f.write(item.result.raw_response_text or "")
            _replace_with_retries(temp_log, log_path)
        finally:
            try:
                if temp_log.exists():
                    temp_log.unlink()
            except Exception:
                pass

        item.local_integrity.output_written = True
        item.local_integrity.log_written = True
        item.processing.status = ItemStatus.APPLIED.value
        item.processing.status_reason = "Keywords written to image/log"
        item.processing.applied_at = utc_now_iso()
        item.processing.error_reason = None

        append_journal(root_dir, {
            "time": utc_now_iso(),
            "custom_id": item.custom_id,
            "action": "applied",
            "output": item.output_relpath,
            "log": item.log_relpath,
        })
        return True
    except Exception as e:
        try:
            if 'temp_target' in locals() and temp_target and Path(temp_target).exists():
                Path(temp_target).unlink()
        except Exception:
            pass
        try:
            if 'temp_log' in locals() and temp_log and Path(temp_log).exists():
                Path(temp_log).unlink()
        except Exception:
            pass
        item.processing.status = ItemStatus.FAILED_APPLY.value
        item.processing.status_reason = f"Apply failed: {e}"
        item.processing.failed_at = utc_now_iso()
        item.processing.error_reason = str(e)
        append_journal(root_dir, {
            "time": utc_now_iso(),
            "custom_id": item.custom_id,
            "action": "apply_failed",
            "reason": str(e),
        })
        return False


def apply_all_ready_items(root_dir: Path, meta: BatchMeta, manifest: Manifest, *, verbose: bool = False) -> Tuple[int, int]:
    applied = 0
    failed = 0
    meta.flags.apply_started = True
    ready = [
        item for item in manifest.items
        if item.processing.status == ItemStatus.COMPLETED.value
        and (not meta.active_batch_id or item.processing.batch_id == meta.active_batch_id)
    ]
    if ready:
        print(f"\n💾 APPLYING {len(ready)} result(s) to images")
        print("─" * 70)
    for idx, item in enumerate(ready, 1):
        print(f"  [{idx}/{len(ready)}] {Path(item.source_relpath).name}", end="", flush=True)
        ok = apply_completed_item(root_dir, meta, item, verbose=verbose)
        if ok:
            applied += 1
            print(f"  ✅")
        else:
            failed += 1
            print(f"  ❌ {item.processing.error_reason or 'unknown error'}")
    if ready:
        print("─" * 70)
        print(f"  Applied: {applied}  |  Failed: {failed}")
    sync_meta_from_manifest(meta, manifest)
    save_manifest(root_dir, manifest)
    save_meta(root_dir, meta)
    return applied, failed


# ---------------------------------------------------------------------------
# Batch lifecycle orchestration
# ---------------------------------------------------------------------------


def clear_active_batch_tracking(meta: BatchMeta) -> None:
    meta.active_batch_id = None
    meta.active_input_file_id = None
    meta.active_output_file_id = None
    meta.active_error_file_id = None
    meta.remote_status = None
    meta.current_chunk_bytes = 0
    meta.flags.requests_uploaded = False
    meta.flags.batch_created = False
    meta.flags.output_downloaded = False
    meta.flags.error_output_downloaded = False
    meta.flags.apply_started = False
    meta.flags.apply_finished = False


def maybe_archive_completed_jsonl(root_dir: Path, meta: BatchMeta) -> None:
    archive_dir = state_file(root_dir, ARCHIVE_DIRNAME)
    archive_dir.mkdir(parents=True, exist_ok=True)
    chunk = meta.current_chunk_index
    for name in [REQUESTS_JSONL, OUTPUT_JSONL, ERROR_OUTPUT_JSONL]:
        p = state_file(root_dir, name)
        if p.exists():
            dest = archive_dir / f"chunk_{chunk:03d}_{name}"
            if dest.exists():
                dest.unlink()
            p.replace(dest)


def finalize_finished_batch(root_dir: Path, meta: BatchMeta, manifest: Manifest) -> None:
    unresolved = unresolved_active_items(manifest, meta.active_batch_id)
    if unresolved:
        return
    maybe_archive_completed_jsonl(root_dir, meta)
    clear_active_batch_tracking(meta)
    sync_meta_from_manifest(meta, manifest)
    save_manifest(root_dir, manifest)
    save_meta(root_dir, meta)


def start_new_batch_if_possible(root_dir: Path, meta: BatchMeta, manifest: Manifest, config: APIConfig, *, verbose: bool = False) -> bool:
    if meta.active_batch_id:
        return False
    if not any_pending_local(manifest):
        return False
    selected = prepare_next_chunk(root_dir, meta, manifest, config, verbose=verbose)
    if selected <= 0:
        return False
    submit_prepared_chunk(root_dir, meta, manifest, config)
    return True


def poll_and_progress(root_dir: Path, meta: BatchMeta, manifest: Manifest, config: APIConfig, *, verbose: bool = False) -> dict:
    result: Dict[str, Any] = {}
    if not meta.active_batch_id:
        created = start_new_batch_if_possible(root_dir, meta, manifest, config, verbose=verbose)
        sync_meta_from_manifest(meta, manifest)
        save_manifest(root_dir, manifest)
        save_meta(root_dir, meta)
        if not created:
            print("  ℹ️  Nothing to do — no pending images and no active batch.")
        return {"action": "submitted_new_batch" if created else "nothing_to_do", "batch_id": meta.active_batch_id}

    print(f"\n🔄 Polling batch {meta.active_batch_id}...", end="", flush=True)
    batch = poll_active_batch(root_dir, meta, config)
    counts = batch.get("request_counts") or {}
    status = batch.get("status", "unknown")
    c_done = counts.get("completed", "?")
    c_fail = counts.get("failed", "?")
    c_total = counts.get("total", "?")
    print(f"  status: {status}  (completed={c_done}, failed={c_fail}, total={c_total})")
    result["remote_status"] = status
    result["batch_id"] = meta.active_batch_id

    if batch.get("status") in {"completed", "cancelled"}:
        download_active_outputs(root_dir, meta, config)
        completed, failed_parse = ingest_output_jsonl(root_dir, meta, manifest)
        failed_remote = ingest_error_jsonl(root_dir, meta, manifest)
        applied, failed_apply = apply_all_ready_items(root_dir, meta, manifest, verbose=verbose)
        result.update({
            "completed_results": completed,
            "failed_parse": failed_parse,
            "failed_remote": failed_remote,
            "applied": applied,
            "failed_apply": failed_apply,
        })
        finalize_finished_batch(root_dir, meta, manifest)
        if not meta.active_batch_id and any_pending_local(manifest):
            result["next_chunk_available"] = True
        sync_meta_from_manifest(meta, manifest)
        save_manifest(root_dir, manifest)
        save_meta(root_dir, meta)
    elif batch.get("status") in {"failed", "expired"}:
        # Mark still-submitted items in this batch as remote failures.
        for item in manifest.items:
            if item.processing.batch_id == meta.active_batch_id and item.processing.status in {ItemStatus.SUBMITTED.value, ItemStatus.PREPARED.value}:
                item.processing.status = ItemStatus.FAILED_REMOTE.value
                item.processing.status_reason = f"Batch ended with status {batch.get('status')}"
                item.processing.failed_at = utc_now_iso()
                item.processing.error_reason = batch.get("status")
        finalize_finished_batch(root_dir, meta, manifest)
        sync_meta_from_manifest(meta, manifest)
        save_manifest(root_dir, manifest)
        save_meta(root_dir, meta)
        result["terminal_error"] = batch.get("status")
    else:
        sync_meta_from_manifest(meta, manifest)
        save_meta(root_dir, meta)
    return result


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def print_summary(meta: BatchMeta, manifest: Optional[Manifest] = None) -> None:
    c = meta.counters
    print("\n" + "=" * 70)
    print("BATCH IMAGETAGGER STATUS")
    print("=" * 70)
    print(f"Directory: {meta.directory}")
    if meta.settings:
        print(f"Model:     {meta.settings.model}")
        print(f"Marker:    {meta.settings.marker}")
        print(f"Mode:      {'overwrite originals' if meta.settings.overwrite else 'create enriched copies'}")
        print(f"Marker field: {'XPComment' if meta.settings.use_xpcomment else 'XPKeywords'}")
        print(f"Batch cap: {meta.settings.max_batch_bytes} bytes")
    print(f"Job state:  {meta.job_state}")
    print(f"Reason:     {meta.job_state_reason}")
    if meta.active_batch_id:
        print(f"Active batch: {meta.active_batch_id}")
        print(f"Remote status: {meta.remote_status}")
    if meta.last_poll_time:
        print(f"Last poll:  {meta.last_poll_time}")
        if meta.last_poll_note:
            print(f"Poll note:  {meta.last_poll_note}")
    print("-" * 70)
    print(f"Total items:    {c.total_items}")
    print(f"Pending local:  {c.pending_local}")
    print(f"Prepared:       {c.prepared}")
    print(f"Submitted:      {c.submitted}")
    print(f"Completed:      {c.completed}")
    print(f"Applied:        {c.applied}")
    print(f"Skipped:        {c.skipped}")
    print(f"Failed remote:  {c.failed_remote}")
    print(f"Failed parse:   {c.failed_parse}")
    print(f"Failed apply:   {c.failed_apply}")
    print(f"Invalidated:    {c.invalidated}")
    if manifest:
        failed_items = [
            item for item in manifest.items
            if item.processing.status in {
                ItemStatus.FAILED_REMOTE.value,
                ItemStatus.FAILED_PARSE.value,
                ItemStatus.FAILED_APPLY.value,
                ItemStatus.INVALIDATED.value,
            }
        ]
        if failed_items:
            print("-" * 70)
            print("Failed files:")
            for item in failed_items:
                reason = item.processing.error_reason or item.processing.status_reason or item.processing.status
                print(f"  ❌ {item.source_relpath}: {reason[:100]}")
    print("=" * 70)


def cmd_cancel(root_dir: Path, meta: BatchMeta, config: APIConfig) -> None:
    if not meta.active_batch_id:
        print("No active batch to cancel.")
        return
    result = cancel_batch(config, meta.active_batch_id)
    meta.remote_status = result.get("status")
    meta.last_poll_time = utc_now_iso()
    meta.last_poll_note = "Cancel requested"
    save_meta(root_dir, meta)
    print(f"Cancel requested for batch {meta.active_batch_id}. New status: {meta.remote_status}")


def cmd_reset(root_dir: Path, *, yes: bool = False) -> None:
    sd = state_dir(root_dir)
    if not sd.exists():
        print("No state directory exists.")
        return
    if not yes:
        raise RuntimeError("Refusing to reset without --yes")
    archive = sd.parent / f"{STATE_DIRNAME}_reset_{int(time.time())}"
    sd.replace(archive)
    print(f"Moved state directory to: {archive}")


# ---------------------------------------------------------------------------
# Program entry
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batch_imagetagger",
        description="Tag images with AI-generated keywords using the OpenAI Batch API.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-d", "--directory", type=str, default=".", help="Directory containing images to process")
    parser.add_argument("-e", "--env", type=str, default=None, help="Path to environment file with API key")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite original images instead of creating copies")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-f", "--force", action="store_true", help="Ignore already-processed marker checks when building a new manifest")
    parser.add_argument("-t", "--tag", type=str, default=DEFAULT_TAG, help=f"Marker tag (default: {DEFAULT_TAG})")
    parser.add_argument("-x", "--xpcomment", action="store_true", help="Store processing marker only in XPComment")
    parser.add_argument("-m", "--model", type=str, default=None, help="Override the model from env.txt")
    parser.add_argument("-T", "--temperature", type=float, default=0.2, help="AI sampling temperature 0.0–1.0 (default: 0.2)")
    parser.add_argument("--dropgps", action="store_true", help="Strip GPS coordinates from output images")
    parser.add_argument("--dropdatetime", action="store_true", help="Strip datetime fields from output images")
    parser.add_argument("--max-batch-bytes", type=int, default=DEFAULT_MAX_BATCH_BYTES, help=f"Soft cap for one batch input file (default: {DEFAULT_MAX_BATCH_BYTES})")
    parser.add_argument("--status", action="store_true", help="Only show local status, do not poll or submit")
    parser.add_argument("--cancel", action="store_true", help="Cancel the current active remote batch")
    parser.add_argument("--reset", action="store_true", help="Archive local batch state and start over on next run")
    parser.add_argument("--yes", action="store_true", help="Confirm destructive operation such as --reset")
    return parser


def load_or_create_state(root_dir: Path, args, config: APIConfig) -> Tuple[BatchMeta, Manifest, bool]:
    meta_path = state_file(root_dir, BATCH_META_JSON)
    manifest_path = state_file(root_dir, MANIFEST_JSON)
    if meta_path.exists() and manifest_path.exists():
        meta = load_meta(root_dir)
        manifest = load_manifest(root_dir)
        return meta, manifest, False

    settings = BatchSettings(
        model=config.model,
        marker=args.tag,
        use_xpcomment=args.xpcomment,
        overwrite=args.overwrite,
        force=args.force,
        max_batch_bytes=args.max_batch_bytes,
        temperature=args.temperature,
        drop_gps=args.dropgps,
        drop_datetime=args.dropdatetime,
    )
    meta = BatchMeta(directory=str(root_dir), settings=settings)
    manifest = build_manifest(root_dir, settings, verbose=args.verbose)
    sync_meta_from_manifest(meta, manifest)
    save_manifest(root_dir, manifest)
    save_meta(root_dir, meta)
    return meta, manifest, True


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    root_dir = Path(args.directory).resolve()
    if not root_dir.is_dir():
        print(f"Error: {root_dir} is not a directory.")
        return 1

    if args.reset:
        try:
            cmd_reset(root_dir, yes=args.yes)
        except Exception as e:
            print(f"Reset failed: {e}")
            return 1
        return 0

    try:
        config = APIConfig(args.env)
        config.apply_overrides(model_override=args.model)
    except Exception as e:
        print(f"Config error: {e}")
        return 1

    try:
        print(f"\n{'='*70}")
        print("BATCH IMAGETAGGER - AI Image Keyword Extractor")
        print(f"{'='*70}")
        print(f"Model:    {config.model}")
        print(f"Input:    {root_dir}")
        print(f"Mode:     {'Overwrite originals' if args.overwrite else 'Create enriched copies'}")
        print(f"Marker:   '{args.tag}'  ({'XPComment' if args.xpcomment else 'XPKeywords'})")
        print(f"Temp:     {args.temperature}")
        privacy = []
        if args.dropgps:
            privacy.append("GPS stripped")
        if args.dropdatetime:
            privacy.append("datetime stripped")
        if privacy:
            print(f"Privacy:  {', '.join(privacy)}")
        print(f"{'='*70}")

        meta, manifest, created = load_or_create_state(root_dir, args, config)
        if created:
            print(f"\n✅ Created new batch state in {state_dir(root_dir)}")

        # Existing state wins over fresh CLI flags for ongoing work.
        if meta.settings:
            if args.max_batch_bytes and args.max_batch_bytes != meta.settings.max_batch_bytes:
                meta.settings.max_batch_bytes = args.max_batch_bytes
                save_meta(root_dir, meta)

        sync_meta_from_manifest(meta, manifest)
        save_meta(root_dir, meta)
        save_manifest(root_dir, manifest)

        if args.status:
            print_summary(meta, manifest)
            return 0

        if args.cancel:
            cmd_cancel(root_dir, meta, config)
            print_summary(load_meta(root_dir), load_manifest(root_dir))
            return 0

        result = poll_and_progress(root_dir, meta, manifest, config, verbose=args.verbose)
        meta = load_meta(root_dir)
        manifest = load_manifest(root_dir)
        print_summary(meta, manifest)
        if args.verbose and result:
            print("Result detail:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
