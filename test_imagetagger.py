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
    APIConfig,
    parse_keywords,
    parse_ratelimit_reset,
    strip_thinking,
    sanitize_keywords,
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

# --- Keyword sanitization ---

@pytest.mark.parametrize("raw, expected", [
    # List markers stripped
    (["- cat", "* dog", "• bird"], ["cat", "dog", "bird"]),
    # Numbering stripped
    (["1. cat", "2) dog"], ["cat", "dog"]),
    # Trailing punctuation stripped
    (["cat.", "dog,", "bird;"], ["cat", "dog", "bird"]),
    # Duplicates (case-insensitive) removed
    (["Cat", "cat", "CAT"], ["Cat"]),
    # Filler words rejected
    (["and", "the", "a", "cat"], ["cat"]),
    # Too long (>40 chars) rejected
    (["x" * 41, "cat"], ["cat"]),
    # Too many words (>4) rejected
    (["one two three four five", "cat"], ["cat"]),
    # Prose punctuation (internal) rejected
    (["nice photo. great shot", "cat"], ["cat"]),
    # Quotes stripped
    (['"cat"', "'dog'"], ["cat", "dog"]),
    # max_keywords respected
    ([f"word{i}" for i in range(30)], [f"word{i}" for i in range(20)]),
])
def test_sanitize_keywords(raw, expected):
    assert sanitize_keywords(raw) == expected


# --- Thinking stripping ---

@pytest.mark.parametrize("content, expected", [
    ("<think>Let me think...</think>cat, dog", "cat, dog"),
    ("<think>\nlong\nmultiline\nthinking\n</think>\ncat, dog", "cat, dog"),
    ("cat, dog", "cat, dog"),  # no thinking block — unchanged
    ("<think>nested</think>  cat  ", "cat"),  # whitespace stripped
])
def test_strip_thinking(content, expected):
    assert strip_thinking(content) == expected


@patch('imagetagger.time.sleep')
@patch('imagetagger.requests.post')
def test_thinking_stripped_before_keywords(mock_post, mock_sleep, dummy_image, mock_config):
    """<think> blocks in the API response are removed before keyword parsing"""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {}
    mock_resp.json.return_value = {
        'choices': [{'message': {'content': '<think>reasoning here</think>cat, dog, tree'}}]
    }
    mock_post.return_value = mock_resp

    b64, _, _ = resize_for_api(dummy_image)
    raw, keywords, balance = imagetagger.call_AI_vision_for_keywords(b64, "", mock_config)

    assert raw == "cat, dog, tree"
    assert keywords == ["cat", "dog", "tree"]


@patch('imagetagger.time.sleep')
@patch('imagetagger.requests.post')
def test_venice_parameters_sent_for_venice_api(mock_post, mock_sleep, dummy_image, mock_config):
    """venice_parameters block is included in payload when is_venice is True"""
    mock_config.is_venice = True
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {}
    mock_resp.json.return_value = {'choices': [{'message': {'content': 'cat'}}]}
    mock_post.return_value = mock_resp

    b64, _, _ = resize_for_api(dummy_image)
    imagetagger.call_AI_vision_for_keywords(b64, "", mock_config)

    payload = mock_post.call_args[1]['json']
    assert 'venice_parameters' in payload
    assert payload['venice_parameters']['include_venice_system_prompt'] is False
    assert payload['venice_parameters']['disable_thinking'] is True


@patch('imagetagger.time.sleep')
@patch('imagetagger.requests.post')
def test_venice_parameters_not_sent_for_other_api(mock_post, mock_sleep, dummy_image, mock_config):
    """venice_parameters block is NOT included when is_venice is False"""
    mock_config.is_venice = False
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {}
    mock_resp.json.return_value = {'choices': [{'message': {'content': 'cat'}}]}
    mock_post.return_value = mock_resp

    b64, _, _ = resize_for_api(dummy_image)
    imagetagger.call_AI_vision_for_keywords(b64, "", mock_config)

    payload = mock_post.call_args[1]['json']
    assert 'venice_parameters' not in payload


# --- Rate-limit reset header parsing ---

@pytest.mark.parametrize("value, expected", [
    ("1m4.637s", 64.637),
    ("3m26.938s", 206.938),
    ("30s", 30.0),
    ("500ms", 0.5),
    ("1ms", 0.001),
    ("2m0s", 120.0),
    (None, None),
    ("", None),
])
def test_parse_ratelimit_reset(value, expected):
    result = parse_ratelimit_reset(value)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected, rel=1e-4)


@patch('imagetagger.time.sleep')
@patch('imagetagger.requests.post')
def test_429_uses_header_reset_time(mock_post, mock_sleep, dummy_image, mock_config):
    """On 429 with tokens exhausted, wait time comes from x-ratelimit-reset-tokens header"""
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.headers = {
        'x-ratelimit-remaining-tokens': '0',
        'x-ratelimit-reset-tokens': '1m4.637s',
        'x-ratelimit-remaining-requests': '9975',
        'x-ratelimit-reset-requests': '3m26.938s',
    }

    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.headers = {}
    mock_resp_200.json.return_value = {'choices': [{'message': {'content': 'ok'}}]}

    mock_post.side_effect = [mock_resp_429, mock_resp_200]

    b64, _, _ = resize_for_api(dummy_image)
    imagetagger.call_AI_vision_for_keywords(b64, "", mock_config)

    # The retry sleep should be ~64.637s (from the header), not the default 1s backoff
    retry_sleeps = [c.args[0] for c in mock_sleep.call_args_list if c.args[0] >= 1]
    assert any(abs(s - 64.637) < 0.01 for s in retry_sleeps)


@patch('imagetagger.time.sleep')
@patch('imagetagger.requests.post')
def test_429_fallback_to_backoff_without_headers(mock_post, mock_sleep, dummy_image, mock_config):
    """On 429 without usable headers, falls back to exponential backoff capped at 16s"""
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.headers = {}  # no rate limit headers

    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.headers = {}
    mock_resp_200.json.return_value = {'choices': [{'message': {'content': 'ok'}}]}

    mock_post.side_effect = [mock_resp_429, mock_resp_200]

    b64, _, _ = resize_for_api(dummy_image)
    imagetagger.call_AI_vision_for_keywords(b64, "", mock_config)

    retry_sleeps = [c.args[0] for c in mock_sleep.call_args_list if c.args[0] >= 1]
    assert retry_sleeps[0] == 1  # first attempt: 1s backoff


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

@pytest.fixture
def dummy_png(temp_dir):
    img_path = temp_dir / "test_img.png"
    img = Image.new('RGB', (100, 100), color='blue')
    img.save(img_path)
    return img_path

@pytest.mark.skipif(not PIEXIF_INSTALLED, reason="piexif not installed")
def test_png_write_and_check_xpkeywords(dummy_png):
    """Keywords and marker must be written to and read from PNG EXIF"""
    success = save_with_new_metadata(
        dummy_png, dummy_png,
        keywords=["forest", "nature"], ai_raw_response="forest, nature",
        marker="jms", use_xpcomment=False
    )
    assert success is True
    found = check_already_processed(dummy_png, "jms", use_xpcomment=False)
    assert found is True

@pytest.mark.skipif(not PIEXIF_INSTALLED, reason="piexif not installed")
def test_png_write_and_check_xpcomment(dummy_png):
    """Marker stored in XPComment must survive a PNG round-trip"""
    success = save_with_new_metadata(
        dummy_png, dummy_png,
        keywords=["sky"], ai_raw_response="sky",
        marker="jms", use_xpcomment=True
    )
    assert success is True
    found = check_already_processed(dummy_png, "jms", use_xpcomment=True)
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
    config = APIConfig(env_file_path=env_file)
    assert config.api_key == "TEST_API_KEY"
    assert config.model == "test-model"

def test_vision_detection(temp_dir):
    env_path = temp_dir / "env_vision.txt"
    env_path.write_text("api_key=key\nmodel=llava-vision")
    
    config = APIConfig(env_file_path=env_path)
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
    raw, keywords, balance = imagetagger.call_AI_vision_for_keywords(b64, "", mock_config)

    assert keywords == ['sky', 'cloud', 'blue']
    assert raw == 'sky, cloud, blue'

@patch('imagetagger.time.sleep')
@patch('imagetagger.requests.post')
def test_api_rate_limit_retry(mock_post, mock_sleep, dummy_image, mock_config):
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
    raw, keywords, balance = imagetagger.call_AI_vision_for_keywords(b64, "", mock_config)

    # Should have retried once
    assert mock_post.call_count == 2
    assert keywords == ['retry success']


@patch('imagetagger.time.sleep')
@patch('imagetagger.requests.post')
def test_rate_limit_up_to_10_retries(mock_post, mock_sleep, dummy_image, mock_config):
    """All 10 attempts return 429 -> ERROR: Max retries exceeded"""
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.headers = {}
    mock_post.return_value = mock_resp_429

    b64, _, _ = resize_for_api(dummy_image)
    raw, keywords, balance = imagetagger.call_AI_vision_for_keywords(b64, "", mock_config)

    assert mock_post.call_count == 10
    assert raw == "ERROR: Max retries exceeded"
    assert keywords == []


@patch('imagetagger.time.sleep')
@patch('imagetagger.requests.post')
def test_retry_delay_capped_at_16s(mock_post, mock_sleep, dummy_image, mock_config):
    """Retry delays double each attempt but must not exceed 16s"""
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.headers = {}
    mock_post.return_value = mock_resp_429

    b64, _, _ = resize_for_api(dummy_image)
    imagetagger.call_AI_vision_for_keywords(b64, "", mock_config)

    # sleep is called for throttle + retry delay each attempt
    retry_sleeps = [c.args[0] for c in mock_sleep.call_args_list if c.args[0] >= 1]
    assert all(s <= 16 for s in retry_sleeps)
    assert max(retry_sleeps) == 16


@patch('imagetagger.time.sleep')
@patch('imagetagger.requests.post')
def test_throttle_delay_increases_on_429(mock_post, mock_sleep, dummy_image, mock_config):
    """Adaptive throttle delay grows by 10% on each 429"""
    imagetagger._throttle_delay = 0.1

    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.headers = {}

    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.headers = {}
    mock_resp_200.json.return_value = {'choices': [{'message': {'content': 'ok'}}]}

    mock_post.side_effect = [mock_resp_429, mock_resp_429, mock_resp_200]

    b64, _, _ = resize_for_api(dummy_image)
    imagetagger.call_AI_vision_for_keywords(b64, "", mock_config)

    assert imagetagger._throttle_delay == pytest.approx(0.1 * 1.1 * 1.1 * 0.98, rel=1e-6)


@patch('imagetagger.time.sleep')
@patch('imagetagger.requests.post')
def test_throttle_delay_decreases_on_success(mock_post, mock_sleep, dummy_image, mock_config):
    """Adaptive throttle delay shrinks by 2% on success, floored at 0.1"""
    imagetagger._throttle_delay = 0.5

    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.headers = {}
    mock_resp_200.json.return_value = {'choices': [{'message': {'content': 'ok'}}]}
    mock_post.return_value = mock_resp_200

    b64, _, _ = resize_for_api(dummy_image)
    imagetagger.call_AI_vision_for_keywords(b64, "", mock_config)

    assert imagetagger._throttle_delay == pytest.approx(0.5 * 0.98, rel=1e-6)


@patch('imagetagger.time.sleep')
@patch('imagetagger.requests.post')
def test_throttle_delay_floored_at_0_1(mock_post, mock_sleep, dummy_image, mock_config):
    """Throttle delay never drops below 0.1s"""
    imagetagger._throttle_delay = 0.1

    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.headers = {}
    mock_resp_200.json.return_value = {'choices': [{'message': {'content': 'ok'}}]}
    mock_post.return_value = mock_resp_200

    b64, _, _ = resize_for_api(dummy_image)
    imagetagger.call_AI_vision_for_keywords(b64, "", mock_config)

    assert imagetagger._throttle_delay == pytest.approx(0.1, rel=1e-6)

# --- Workflow Tests ---

@patch('imagetagger.call_AI_vision_for_keywords')
@patch('imagetagger.APIConfig')
def test_skip_logic(MockConfig, mock_call_keywords, temp_dir, dummy_image, env_file):
    """Test that force=False skips already processed images"""
    
    # Setup Mock Config
    cfg = MagicMock()
    cfg.model = "test"
    cfg.is_vision = True
    MockConfig.return_value = cfg
    
    # Setup Mock API Response
    mock_call_keywords.return_value = ("tag1, tag2", ["tag1", "tag2"], None)

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

@patch('imagetagger.call_AI_vision_for_keywords')
@patch('imagetagger.APIConfig')
def test_force_logic(MockConfig, mock_call_keywords, temp_dir, dummy_image, env_file):
    """Test that force=True processes images again"""
    
    cfg = MagicMock()
    cfg.model = "test"
    cfg.is_vision = True
    MockConfig.return_value = cfg
    mock_call_keywords.return_value = ("tag1", ["tag1"], None)

    # 1. First Run
    process_images(str(temp_dir), force=False, env_file=env_file, marker="jms")
    mock_call_keywords.reset_mock()

    # 2. Second Run with Force
    process_images(str(temp_dir), force=True, env_file=env_file, marker="jms")

    # Verify API WAS called again
    assert mock_call_keywords.called

@patch('imagetagger.call_AI_vision_for_keywords')
@patch('imagetagger.APIConfig')
def test_abort_on_invalid_api_key(MockConfig, mock_call, temp_dir, dummy_image, env_file):
    """Process should abort immediately on 401 Invalid API key, not just skip the image"""
    cfg = MagicMock()
    cfg.model = "test"
    cfg.is_vision = True
    MockConfig.return_value = cfg

    mock_call.return_value = ("ERROR_401: Invalid API key", [], None)

    process_images(str(temp_dir), force=True, env_file=env_file)

    # API called only once — aborted, did not continue to next image
    assert mock_call.call_count == 1


@patch('imagetagger.call_AI_vision_for_keywords')
@patch('imagetagger.APIConfig')
def test_abort_on_low_balance(MockConfig, mock_call, temp_dir, dummy_image, env_file):
    """Processing must stop immediately when balance drops below $0.10"""
    cfg = MagicMock()
    cfg.model = "test"
    cfg.is_vision = True
    MockConfig.return_value = cfg

    mock_call.return_value = ("cat, dog", ["cat", "dog"], 0.05)

    process_images(str(temp_dir), force=True, env_file=env_file)

    assert mock_call.call_count == 1  # aborted after first image


@patch('imagetagger.time.sleep')
@patch('imagetagger.call_AI_vision_for_keywords')
@patch('imagetagger.APIConfig')
def test_slowdown_on_medium_low_balance(MockConfig, mock_call, mock_sleep, temp_dir, dummy_image, env_file):
    """A 60s extra sleep must be inserted when balance is between $0.10 and $0.50"""
    cfg = MagicMock()
    cfg.model = "test"
    cfg.is_vision = True
    MockConfig.return_value = cfg

    mock_call.return_value = ("cat, dog", ["cat", "dog"], 0.30)

    process_images(str(temp_dir), force=True, env_file=env_file)

    assert any(c.args[0] == 60 for c in mock_sleep.call_args_list)


@patch('imagetagger.time.sleep')
@patch('imagetagger.call_AI_vision_for_keywords')
@patch('imagetagger.APIConfig')
def test_warning_only_on_low_balance(MockConfig, mock_call, mock_sleep, temp_dir, dummy_image, env_file, capsys):
    """Only a warning is printed (no sleep) when balance is between $0.50 and $1.00"""
    cfg = MagicMock()
    cfg.model = "test"
    cfg.is_vision = True
    MockConfig.return_value = cfg

    mock_call.return_value = ("cat, dog", ["cat", "dog"], 0.75)

    process_images(str(temp_dir), force=True, env_file=env_file)

    assert not any(c.args[0] == 60 for c in mock_sleep.call_args_list)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "0.7500" in captured.out


@patch('imagetagger.save_with_new_metadata')
@patch('imagetagger.call_AI_vision_for_keywords')
@patch('imagetagger.APIConfig')
def test_model_name_appended_to_keywords(MockConfig, mock_call, mock_save, temp_dir, dummy_image, env_file):
    """Model name should be appended to keywords before saving to metadata"""
    cfg = MagicMock()
    cfg.model = "gpt-4o-mini"
    cfg.is_vision = True
    MockConfig.return_value = cfg

    mock_call.return_value = ("cat, dog", ["cat", "dog"], None)
    mock_save.return_value = True

    process_images(str(temp_dir), force=True, env_file=env_file)

    assert mock_save.called
    saved_keywords = mock_save.call_args[0][2]  # 3rd positional arg
    assert "gpt-4o-mini" in saved_keywords
    assert "cat" in saved_keywords
    assert "dog" in saved_keywords


class TestSafetyFeatures:
    """Tests to ensure metadata is never corrupted by errors or refusals"""

    @patch('imagetagger.save_with_new_metadata')
    @patch('imagetagger.call_AI_vision_for_keywords')
    @patch('imagetagger.APIConfig')
    def test_no_write_on_api_error(self, MockConfig, mock_call, mock_save, temp_dir, dummy_image, env_file):
        """Ensure metadata is NOT written if API returns an error code"""
        # Setup Mocks
        cfg = MagicMock()
        cfg.model = "test"
        cfg.is_vision = True
        MockConfig.return_value = cfg
        
        # Simulate API Error
        mock_call.return_value = ("ERROR_401: Invalid API key", [], None)

        # Run
        process_images(str(temp_dir), force=True, env_file=env_file)

        # Assert save function was NEVER called
        mock_save.assert_not_called()

    @patch('imagetagger.save_with_new_metadata')
    @patch('imagetagger.call_AI_vision_for_keywords')
    @patch('imagetagger.APIConfig')
    def test_no_write_on_content_refusal(self, MockConfig, mock_call, mock_save, temp_dir, dummy_image, env_file):
        """Ensure metadata is NOT written if AI refuses content (puritan filter)"""
        # Setup Mocks
        cfg = MagicMock()
        cfg.model = "test"
        cfg.is_vision = True
        MockConfig.return_value = cfg
        
        # Simulate Content Refusal
        mock_call.return_value = ("I'm sorry, but I can't assist with that request.", [], None)

        # Run
        process_images(str(temp_dir), force=True, env_file=env_file)

        # Assert save function was NEVER called
        mock_save.assert_not_called()