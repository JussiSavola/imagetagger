import pytest
import os
import sys
import json
import base64
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image
import importlib.util


# Import the module to test
import imagetagger
from imagetagger import (
    VeniceConfig, 
    parse_keywords, 
    resize_for_api, 
    extract_metadata,
    check_already_processed,
    save_with_new_metadata,
    process_images,
    decode_piexif_value
)

PIEXIF_INSTALLED = importlib.util.find_spec("piexif") is not None

# --- Fixtures ---

@pytest.fixture
def temp_dir(tmp_path):
    """Wrapper for pytest's built-in tmp_path"""
    return tmp_path

@pytest.fixture
def dummy_image(temp_dir):
    """Creates a dummy image for testing"""
    img_path = temp_dir / "test_img.jpg"
    img = Image.new('RGB', (1000, 500), color='red')
    img.save(img_path)
    return img_path

@pytest.fixture
def env_file(temp_dir):
    """Creates a dummy env.txt file"""
    env_path = temp_dir / "env.txt"
    env_path.write_text("api_key=TEST_API_KEY\nmodel=test-model")
    return env_path

# --- Utility Tests ---

@pytest.mark.parametrize("response, expected", [
    ("cat, dog, tree, sky", ['cat', 'dog', 'tree', 'sky']),
    ("Here are the keywords: car, bus, train.", ['car', 'bus', 'train']),
    ("apple\nbanana\norange", ['apple', 'banana', 'orange']),
])
def test_parse_keywords_various_inputs(response, expected):
    assert parse_keywords(response) == expected

def test_parse_keywords_limit():
    response = ", ".join([f"word{i}" for i in range(30)])
    keywords = parse_keywords(response)
    assert len(keywords) == 20

# --- Piexif Decoding Tests ---

@pytest.mark.parametrize("input_val, expected", [
    # Tuple input (common piexif output for XP fields) -> "test"
    ((116, 0, 101, 0, 115, 0, 116, 0), "test"),
    # Bytes input -> "test"
    (b't\x00e\x00s\x00t\x00', "test"),
    # None input -> empty string
    (None, ""),
])
def test_decode_piexif_value(input_val, expected):
    assert decode_piexif_value(input_val) == expected

# --- Image Processing Tests ---

def test_resize_for_api(dummy_image):
    b64, orig_sz, new_sz = resize_for_api(dummy_image)
    
    assert orig_sz == (1000, 500)
    # Max dim (800, 600). 1000 width -> scaled to 800 width.
    # Aspect ratio 2:1 -> 800x400.
    assert new_sz == (800, 400)
    assert isinstance(b64, str)

def test_extract_metadata_basic(dummy_image):
    meta = extract_metadata(dummy_image)
    assert "Format: JPEG" in meta['display_lines'][0]
    assert meta['raw_exif'] == {}  # New image has no EXIF

# --- Metadata RW Tests ---

@pytest.mark.skipif(not PIEXIF_INSTALLED, reason="piexif not installed")
def test_write_and_check_xpkeywords(dummy_image):
    # Write marker to XPKeywords (default mode)
    success = save_with_new_metadata(
        dummy_image, dummy_image, 
        keywords=["test"], ai_raw_response="test", 
        marker="jms", use_xpcomment=False
    )
    assert success is True
    
    # Check detection
    found = check_already_processed(dummy_image, "jms", use_xpcomment=False)
    assert found is True

@pytest.mark.skipif(not PIEXIF_INSTALLED, reason="piexif not installed")
def test_write_and_check_xpcomment(dummy_image):
    # Write marker to XPComment (-x mode)
    success = save_with_new_metadata(
        dummy_image, dummy_image, 
        keywords=["test"], ai_raw_response="test", 
        marker="jms", use_xpcomment=True
    )
    assert success is True
    
    # Check detection
    found = check_already_processed(dummy_image, "jms", use_xpcomment=True)
    assert found is True

@pytest.mark.skipif(not PIEXIF_INSTALLED, reason="piexif not installed")
def test_check_negative(dummy_image):
    # Fresh image, no marker
    found = check_already_processed(dummy_image, "jms", use_xpcomment=False)
    assert found is False

@pytest.mark.skipif(not PIEXIF_INSTALLED, reason="piexif not installed")
def test_check_wrong_marker(dummy_image):
    # Write 'jms'
    save_with_new_metadata(dummy_image, dummy_image, [], "", marker="jms", use_xpcomment=False)
    # Look for 'other'
    found = check_already_processed(dummy_image, "other", use_xpcomment=False)
    assert found is False

# --- Config Tests ---

def test_config_loading(env_file):
    config = VeniceConfig(env_file_path=env_file)
    assert config.api_key == "TEST_API_KEY"
    assert config.model == "test-model"

def test_vision_detection(temp_dir):
    env_path = temp_dir / "env_vision.txt"
    env_path.write_text("api_key=key\nmodel=llava-vision")
    
    config = VeniceConfig(env_file_path=env_path)
    assert config.is_vision is True

# --- API Mocking Tests ---

@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.model = "mock-model"
    cfg.base_url = "http://mock.api"
    cfg.get_headers.return_value = {}
    return cfg

@patch('imagetagger.requests.post')
def test_api_call_success(mock_post, dummy_image, mock_config):
    # Mock Response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.json.return_value = {
        'choices': [{'message': {'content': 'sky, cloud, blue'}}]
    }
    mock_post.return_value = mock_response

    b64, _, _ = resize_for_api(dummy_image)
    raw, keywords = imagetagger.call_venice_for_keywords(b64, "", mock_config)

    assert keywords == ['sky', 'cloud', 'blue']
    assert raw == 'sky, cloud, blue'

@patch('imagetagger.requests.post')
def test_api_rate_limit_retry(mock_post, dummy_image, mock_config):
    # Mock Responses: 429 then 200
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.headers = {}

    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.headers = {}
    mock_resp_200.json.return_value = {'choices': [{'message': {'content': 'retry success'}}]}

    mock_post.side_effect = [mock_resp_429, mock_resp_200]

    b64, _, _ = resize_for_api(dummy_image)
    raw, keywords = imagetagger.call_venice_for_keywords(b64, "", mock_config)

    # Should have retried once
    assert mock_post.call_count == 2
    assert keywords == ['retry success']

# --- Workflow Tests ---

@patch('imagetagger.call_venice_for_keywords')
@patch('imagetagger.VeniceConfig')
def test_skip_logic(MockConfig, mock_call_keywords, temp_dir, dummy_image, env_file):
    """Test that force=False skips already processed images"""
    
    # Setup Mock Config
    cfg = MagicMock()
    cfg.model = "test"
    cfg.is_vision = True
    MockConfig.return_value = cfg
    
    # Setup Mock API Response
    mock_call_keywords.return_value = ("tag1, tag2", ["tag1", "tag2"])

    # 1. First Run
    process_images(
        str(temp_dir), 
        overwrite=False, 
        force=False, 
        env_file=env_file,
        marker="jms"
    )
    
    # Verify API was called
    assert mock_call_keywords.called
    mock_call_keywords.reset_mock()

    # 2. Second Run (Should Skip)
    process_images(
        str(temp_dir), 
        overwrite=False, 
        force=False, 
        env_file=env_file,
        marker="jms"
    )

    # Verify API was NOT called again
    assert not mock_call_keywords.called

@patch('imagetagger.call_venice_for_keywords')
@patch('imagetagger.VeniceConfig')
def test_force_logic(MockConfig, mock_call_keywords, temp_dir, dummy_image, env_file):
    """Test that force=True processes images again"""
    
    cfg = MagicMock()
    cfg.model = "test"
    cfg.is_vision = True
    MockConfig.return_value = cfg
    mock_call_keywords.return_value = ("tag1", ["tag1"])

    # 1. First Run
    process_images(str(temp_dir), force=False, env_file=env_file, marker="jms")
    mock_call_keywords.reset_mock()

    # 2. Second Run with Force
    process_images(str(temp_dir), force=True, env_file=env_file, marker="jms")

    # Verify API WAS called again
    assert mock_call_keywords.called