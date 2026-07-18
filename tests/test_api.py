import pytest
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlmodel import Session
from database.models import VlogJob

def test_read_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_upload_invalid_file_format(client: TestClient):
    # Try uploading a text file
    files = {"file": ("test.txt", b"some plain text", "text/plain")}
    response = client.post("/api/vlog/upload", files=files)
    assert response.status_code == 400
    assert "Unsupported video format" in response.json()["detail"]

@patch("api.routes.VideoFileClip")
@patch("api.routes.shutil.copyfileobj")
def test_upload_valid_video(mock_copy, mock_video_clip, client: TestClient):
    # Mock moviepy video clip duration
    mock_instance = MagicMock()
    mock_instance.duration = 15.5
    mock_video_clip.return_value = mock_instance

    # Mock upload
    files = {"file": ("holiday_trip.mp4", b"dummy video bytes", "video/mp4")}
    response = client.post("/api/vlog/upload", files=files)
    
    assert response.status_code == 201
    data = response.json()
    assert "video_id" in data
    assert data["video_id"].startswith("holiday_trip--v_")
    assert data["filename"] == "holiday_trip.mp4"
    assert data["duration_seconds"] == 15.5
    
    # Clean up test output if created
    uploaded_path = data["file_path"]
    if uploaded_path and os.path.exists(uploaded_path):
        os.remove(uploaded_path)
@patch("api.routes.BackgroundTasks.add_task")
def test_process_vlog_trigger(mock_add_task, client: TestClient, session: Session):
    # Stash dummy upload in local uploads directory to pass route validation
    from config import settings
    video_id = "v_test123"
    dummy_video_path = settings.UPLOAD_DIR / f"{video_id}.mp4"
    with open(dummy_video_path, "wb") as f:
        f.write(b"dummy mp4")
        
    try:
        request_data = {
            "video_id": video_id,
            "prompt_hint": "Fun times in Paris",
            "music_vibe": "Cinematic & Adventurous",
            "tts_provider": "edge-tts",
            "voice_id": "en-US-GuyNeural"
        }
        
        response = client.post("/api/vlog/process", json=request_data)
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "PENDING"
        
        # Verify background pipeline task was queued
        mock_add_task.assert_called_once()
        
        # Verify job is present in DB
        db_job = session.get(VlogJob, data["job_id"])
        assert db_job is not None
        assert db_job.video_id == video_id
        assert db_job.prompt_hint == "Fun times in Paris"
        
    finally:
        # Clean up
        if dummy_video_path.exists():
            dummy_video_path.unlink()

def test_get_status_nonexistent_job(client: TestClient):
    response = client.get("/api/vlog/status/job_invalid")
    assert response.status_code == 404
    assert "Job ID not found" in response.json()["detail"]

def test_get_models_list(client: TestClient):
    response = client.get("/api/vlog/models")
    assert response.status_code == 200
    data = response.json()
    assert "gemini" in data
    assert "ollama" in data
    assert len(data["gemini"]) > 0
    assert any(m["id"] == "gemini-2.5-flash" for m in data["gemini"])

@patch("api.routes.BackgroundTasks.add_task")
def test_run_existing_job(mock_add_task, client: TestClient, session: Session):
    # Setup a mock job in database
    job = VlogJob(
        id="job_test_existing",
        status="FAILED",
        progress_percentage=40,
        error_message="some error",
        video_id="v_test",
        filename="master_v_test.mp4",
        model_name="mock"
    )
    session.add(job)
    session.commit()

    response = client.post("/api/vlog/process/job_test_existing")
    assert response.status_code == 202
    data = response.json()
    assert data["job_id"] == "job_test_existing"
    assert data["status"] == "PENDING"

    # Verify database fields were reset
    db_job = session.get(VlogJob, "job_test_existing")
    assert db_job.status == "PENDING"
    assert db_job.progress_percentage == 0
    assert db_job.error_message is None
    
    # Verify background task was enqueued
    mock_add_task.assert_called_once()

def test_delete_job(client: TestClient, session: Session):
    # Setup a mock job in database
    job = VlogJob(
        id="job_test_delete",
        status="FAILED",
        progress_percentage=40,
        error_message="some error",
        video_id="v_test",
        filename="master_v_test.mp4",
        model_name="mock"
    )
    session.add(job)
    session.commit()

    # Verify job exists
    assert session.get(VlogJob, "job_test_delete") is not None

    response = client.delete("/api/vlog/jobs/job_test_delete")
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # Verify job is deleted from database
    assert session.get(VlogJob, "job_test_delete") is None
