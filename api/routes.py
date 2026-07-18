import os
import uuid
import shutil
import asyncio
import httpx
import logging
from datetime import datetime
from typing import List
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlmodel import Session, select

logger = logging.getLogger(__name__)

from config import settings
from database.db import get_session
from database.models import VlogJob, VlogSegment
from api.schemas import ProcessVlogRequest, VlogJobResponse
from services.agent import run_vlog_agent
from services.tts import get_tts_adapter
from services.video import VideoPostProcessingService
from moviepy import VideoFileClip

router = APIRouter(prefix="/api/vlog", tags=["vlog"])

# Background pipeline runner
async def process_vlog_pipeline(job_id: str):
    # Retrieve DB session locally inside background thread
    from database.db import engine
    
    with Session(engine) as db:
        job = db.get(VlogJob, job_id)
        if not job:
            logger.warning(f"Job {job_id} not found in database.")
            return
            
        temp_files = []
        try:
            # 1. Update status to ANALYZING
            job.status = "ANALYZING"
            job.progress_percentage = 15
            job.updated_at = datetime.utcnow()
            db.add(job)
            db.commit()
            db.refresh(job)
            
            # Resolve video path with any supported extension
            video_path = None
            for ext in [".mp4", ".mov", ".avi", ".mkv"]:
                p = settings.UPLOAD_DIR / f"{job.video_id}{ext}"
                if p.exists():
                    video_path = str(p)
                    break
            if not video_path:
                raise FileNotFoundError(f"Source video file not found for video ID {job.video_id}")
                
            # 2. Run LangGraph Orchestrator
            agent_result = await run_vlog_agent(
                video_path=video_path,
                prompt_hint=job.prompt_hint,
                music_vibe=job.music_vibe,
                model_name=job.model_name,
                job_id=job.id
            )
            
            if agent_result.get("error") and not agent_result.get("script_segments"):
                raise Exception(f"LangGraph Agent failed: {agent_result.get('error')}")
                
            script_segments = agent_result.get("script_segments", [])
            
            # Save segments to database
            for seg in script_segments:
                db_seg = VlogSegment(
                    job_id=job.id,
                    start_second=seg["start_second"],
                    end_second=seg["end_second"],
                    voiceover_text=seg["voiceover_text"],
                    music_vibe=seg["music_vibe"]
                )
                db.add(db_seg)
            db.commit()
            db.refresh(job)
            
            # 3. Update status to GENERATING_TTS
            job.status = "GENERATING_TTS"
            job.progress_percentage = 50
            job.updated_at = datetime.utcnow()
            db.add(job)
            db.commit()
            
            # Initialize TTS service
            tts_service = get_tts_adapter(job.tts_provider)
            
            voiceover_paths = []
            tts_tasks = []
            
            # Schedule TTS generation concurrently
            for idx, seg in enumerate(script_segments):
                voice_filename = f"voice_{job_id}_{idx}.mp3"
                voice_path = settings.TEMP_DIR / voice_filename
                temp_files.append(str(voice_path))
                voiceover_paths.append(str(voice_path))
                
                # Enqueue TTS generation
                tts_tasks.append(
                    tts_service.generate_audio(
                        text=seg["voiceover_text"],
                        output_path=str(voice_path),
                        voice_id=job.voice_id
                    )
                )
                
            if job.tts_provider.lower() in ("f5-tts", "f5tts", "bark"):
                logger.info(f"[TTS] Running generations sequentially for local provider: {job.tts_provider}")
                tts_results = []
                for task in tts_tasks:
                    res = await task
                    tts_results.append(res)
            else:
                logger.info(f"[TTS] Running generations concurrently for cloud provider: {job.tts_provider}")
                tts_results = await asyncio.gather(*tts_tasks)
            
            if not all(tts_results):
                raise Exception("One or more voiceover TTS generations failed.")
                
            # 4. Update status to ASSEMBLING_VIDEO
            job.status = "ASSEMBLING_VIDEO"
            job.progress_percentage = 80
            job.updated_at = datetime.utcnow()
            db.add(job)
            db.commit()
            
            output_filename = f"master_{job_id}.mp4"
            output_path = settings.OUTPUT_DIR / output_filename
            
            # Gather starts and ends
            segments_timestamps = [(s["start_second"], s["end_second"]) for s in script_segments]
            
            # Render video
            # Assemble video is run in standard thread since MoviePy/FFmpeg write blockingly
            success = await asyncio.to_thread(
                VideoPostProcessingService.assemble_final_video,
                video_path=video_path,
                output_path=str(output_path),
                voiceover_paths=voiceover_paths,
                segments=segments_timestamps,
                music_vibe=job.music_vibe
            )
            
            if not success:
                raise Exception("Video rendering assembly failed.")
                
            # 5. Complete Job
            job.status = "COMPLETED"
            job.progress_percentage = 100
            job.output_video_path = str(output_path)
            job.updated_at = datetime.utcnow()
            db.add(job)
            db.commit()
            logger.info(f"Successfully processed job {job_id}.")
            
        except Exception as e:
            logger.error(f"Crash in job {job_id}: {str(e)}", exc_info=True)
            job.status = "FAILED"
            job.error_message = str(e)
            job.updated_at = datetime.utcnow()
            db.add(job)
            db.commit()
            
        finally:
            # Clean up temporary voice MP3 segments
            for temp_f in temp_files:
                try:
                    if os.path.exists(temp_f):
                        os.remove(temp_f)
                except Exception as ex:
                    logger.error(f"Clean-up error for file {temp_f}: {str(ex)}")

@router.post("/upload", status_code=201)
async def upload_video(file: UploadFile = File(...)):
    """Uploads a raw travel video clip, saves it locally, and registers a video ID."""
    if not file.filename.endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(status_code=400, detail="Unsupported video format. Upload mp4, mov, avi, or mkv.")
        
    original_filename = file.filename
    filename_prefix, ext = os.path.splitext(original_filename)
    
    video_id = f"{filename_prefix}--v_{uuid.uuid4().hex[:8]}"
    destination_path = settings.UPLOAD_DIR / f"{video_id}{ext}"
    
    try:
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Get basic duration check if possible (using MoviePy)
        duration = 0.0
        try:
            clip = VideoFileClip(str(destination_path))
            duration = clip.duration
            clip.close()
        except Exception as e:
            logger.warning(f"Duration extraction failed: {str(e)}")
            
        return {
            "video_id": video_id,
            "filename": file.filename,
            "file_path": str(destination_path),
            "duration_seconds": duration
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/process", status_code=202)
async def process_vlog(
    request: ProcessVlogRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_session)
):
    """Enqueues video log processing in the background and returns job ID."""
    # Check if raw video file exists
    video_path = settings.UPLOAD_DIR / f"{request.video_id}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Source video ID not found.")
        
    # Get filenames
    filename = f"master_{request.video_id}.mp4"
    
    # Create job entry
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    job = VlogJob(
        id=job_id,
        status="PENDING",
        progress_percentage=0,
        video_id=request.video_id,
        filename=filename,
        prompt_hint=request.prompt_hint,
        music_vibe=request.music_vibe,
        tts_provider=request.tts_provider,
        voice_id=request.voice_id,
        model_name=request.model_name
    )
    
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Add background task pipeline running
    background_tasks.add_task(process_vlog_pipeline, job_id)
    
    return {
        "job_id": job_id,
        "status": "PENDING",
        "message": "Vlog processing pipeline initiated in background."
    }

@router.post("/process/{job_id}", status_code=202)
async def run_existing_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_session)
):
    """Triggers/runs an existing pending or failed vlog job."""
    job = db.get(VlogJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found.")
        
    # Reset job fields to process again
    job.status = "PENDING"
    job.progress_percentage = 0
    job.error_message = None
    job.updated_at = datetime.utcnow()
    
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Enqueue background processing
    background_tasks.add_task(process_vlog_pipeline, job_id)
    
    return {
        "job_id": job_id,
        "status": "PENDING",
        "message": f"Job {job_id} processing restarted in background."
    }

@router.delete("/jobs/{job_id}", status_code=200)
async def delete_job(job_id: str, db: Session = Depends(get_session)):
    """Deletes a job and its associated workspace resources (e.g. database entry and output files)."""
    job = db.get(VlogJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found.")
        
    # Delete output video if it exists
    if job.output_video_path and os.path.exists(job.output_video_path):
        try:
            os.remove(job.output_video_path)
        except Exception as e:
            logger.error(f"Failed to delete output video for job {job_id}: {str(e)}")
            
    # Delete temporary synthesized voiceovers
    temp_dir = settings.STORAGE_DIR / "temp"
    if temp_dir.exists():
        for filename in os.listdir(temp_dir):
            if filename.startswith(f"voice_{job_id}"):
                try:
                    os.remove(temp_dir / filename)
                except Exception as e:
                    logger.error(f"Failed to delete temp file {filename}: {str(e)}")
                    
    # Remove from database (cascade deletes segments)
    db.delete(job)
    db.commit()
    
    return {
        "status": "success",
        "message": f"Job {job_id} deleted successfully."
    }

@router.get("/status/{job_id}", response_model=VlogJobResponse)
async def get_job_status(job_id: str, db: Session = Depends(get_session)):
    """Retrieves full job status, logs, and generated script segment timelines."""
    job = db.get(VlogJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    return job

@router.get("/jobs", response_model=List[VlogJobResponse])
async def list_jobs(db: Session = Depends(get_session)):
    """Lists all active and completed vlog generation jobs."""
    jobs = db.exec(select(VlogJob).order_by(VlogJob.created_at.desc())).all()
    return jobs

@router.get("/download/{job_id}")
async def download_vlog(job_id: str, db: Session = Depends(get_session)):
    """Downloads the final stitched video log file."""
    job = db.get(VlogJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found.")
        
    if job.status != "COMPLETED" or not job.output_video_path:
        raise HTTPException(status_code=400, detail="Vlog processing is not complete yet.")
        
    if not os.path.exists(job.output_video_path):
        raise HTTPException(status_code=404, detail="Resulting video file not found on disk.")
        
    return FileResponse(
        path=job.output_video_path,
        media_type="video/mp4",
        filename=f"directed_vlog_{job_id}.mp4"
    )
@router.get("/raw/{video_id}")
async def get_raw_video(video_id: str):
    """Retrieves raw uploaded video stream."""
    video_path = None
    for ext in [".mp4", ".mov", ".avi", ".mkv"]:
        p = settings.UPLOAD_DIR / f"{video_id}{ext}"
        if p.exists():
            video_path = p
            break
    if not video_path:
        raise HTTPException(status_code=404, detail="Raw video ID not found.")
        
    media_type = "video/mp4"
    if video_path.suffix == ".mov":
        media_type = "video/quicktime"
    elif video_path.suffix == ".avi":
        media_type = "video/x-msvideo"
    elif video_path.suffix == ".mkv":
        media_type = "video/x-matroska"
        
    return FileResponse(path=str(video_path), media_type=media_type)

@router.get("/models")
async def get_models():
    """Returns available AI models grouped by provider (Gemini and Ollama)."""
    models = {
        "gemini": [
            {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash (Recommended)"},
            {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash"}
        ],
        "ollama": []
    }
    
    # Fetch local Ollama models dynamically if online
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=3.0)
            if response.status_code == 200:
                data = response.json()
                ollama_models = data.get("models", [])
                for m in ollama_models:
                    name = m.get("name")
                    models["ollama"].append({
                        "id": f"ollama/{name}",
                        "name": f"{name} (Local)"
                    })
    except Exception as e:
        logger.warning(f"Local Ollama API not available: {str(e)}")
        # Default fallbacks if Ollama is not running but we want dropdown suggestions
        models["ollama"] = [
            {"id": "ollama/llama3:latest", "name": "llama3:latest (Local - Offline)"},
            {"id": "ollama/deepseek-r1:8b", "name": "deepseek-r1:8b (Local - Offline)"}
        ]
        
    return models
