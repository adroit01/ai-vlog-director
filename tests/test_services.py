import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock
from services.tts import get_tts_adapter, EdgeTTSAdapter, MockTTSAdapter
from services.search import web_search

@pytest.mark.asyncio
async def test_mock_tts_generation():
    adapter = get_tts_adapter("mock")
    assert isinstance(adapter, MockTTSAdapter)
    
    output_path = "tests/test_silent.mp3"
    try:
        success = await adapter.generate_audio("Hello testing", output_path)
        assert success is True
        assert os.path.exists(output_path)
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)

@pytest.mark.asyncio
@patch("edge_tts.Communicate")
async def test_edge_tts_calls_python_api(mock_communicate):
    # Mock Communicate instance save method
    mock_comm_instance = AsyncMock()
    mock_comm_instance.save.return_value = None
    mock_communicate.return_value = mock_comm_instance
    
    adapter = get_tts_adapter("edge-tts")
    assert isinstance(adapter, EdgeTTSAdapter)
    
    # We patch os.path.exists and getsize to simulate file creation
    with patch("os.path.exists", return_value=True), patch("os.path.getsize", return_value=100):
        success = await adapter.generate_audio("Hello Edge", "fake_path.mp3")
        assert success is True
        
    mock_communicate.assert_called_once_with("Hello Edge", "en-US-GuyNeural")
    mock_comm_instance.save.assert_called_once_with("fake_path.mp3")

@pytest.mark.asyncio
@patch("services.search.DDGS")
async def test_web_search_duckduckgo(mock_ddgs):
    # Mock DDGS instance context manager
    mock_ddg_instance = MagicMock()
    mock_ddg_instance.__enter__.return_value = mock_ddg_instance
    mock_ddg_instance.text.return_value = [
        {"title": "Rohtang Pass scenic route", "body": "Historical high mountain pass on the eastern Pir Panjal Range."}
    ]
    mock_ddgs.return_value = mock_ddg_instance
    
    result = await web_search("Rohtang Pass", max_results=1)
    assert "Pir Panjal" in result
    assert "Source 1" in result

def test_video_description_caching():
    from services.video_description import get_video_description, save_video_description
    import json
    import os
    
    test_video_path = "/path/to/my_trip--v_5dd436c5.mp4"
    file_name = os.path.basename(test_video_path)
    if "--v_" in file_name:
        filename_prefix = file_name.split("--v_")[0]
    else:
        filename_prefix = os.path.splitext(file_name)[0]
        
    assert filename_prefix == "my_trip"
    test_data = {"queries": ["query 1", "query 2"], "visual_summary": "beautiful view"}
    test_json_str = json.dumps(test_data)
    
    try:
        # Verify it initially returns None
        assert get_video_description(filename_prefix) is None
        
        # Save it
        success = save_video_description(filename_prefix, test_json_str)
        assert success is True
        
        # Read it back and verify values
        cached = get_video_description(filename_prefix)
        assert cached is not None
        assert cached["queries"] == ["query 1", "query 2"]
        assert cached["visual_summary"] == "beautiful view"
    finally:
        # Cleanup
        from config import settings
        p = settings.DESCRIPTIONS_DIR / f"{filename_prefix}.json"
        if p.exists():
            p.unlink()

def test_video_post_processing_service_rotation():
    from services.video import VideoPostProcessingService
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.stdout = '{"streams": [{"side_data_list": [{"rotation": 90}]}]}'
        mock_run.return_value = mock_result
        
        rotation = VideoPostProcessingService.get_video_rotation("dummy_path.mp4")
        assert rotation == 90
        mock_run.assert_called_once()

@pytest.mark.asyncio
async def test_missing_gemini_api_key_raises_exception_and_routes_to_error_node():
    from services.agent import run_vlog_agent
    from config import settings
    
    with patch.object(settings, "GEMINI_API_KEY", ""):
        result = await run_vlog_agent(
            video_path="/path/to/uncached_test_video--v_99999999.mp4",
            model_name="gemini-2.5-flash"
        )
        assert result.get("error") is not None
        assert "GEMINI_API_KEY is not set" in result["error"]
        assert result.get("script_segments") == []


