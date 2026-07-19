from typing import List, Optional
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship
from config import settings

class VlogJob(SQLModel, table=True):
    id: str = Field(default=None, primary_key=True)
    status: str = Field(default="PENDING")
    progress_percentage: int = Field(default=0)
    error_message: Optional[str] = Field(default=None)
    video_id: str
    filename: str
    prompt_hint: Optional[str] = Field(default=None)
    music_vibe: str = Field(default="Cinematic & Adventurous")
    tts_provider: str = Field(default="edge-tts")
    voice_id: Optional[str] = Field(default=None)
    model_name: str = Field(default_factory=lambda: settings.DEFAULT_ANALYSIS_MODEL)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    output_video_path: Optional[str] = Field(default=None)
    
    # Relationship to segments
    segments: List["VlogSegment"] = Relationship(back_populates="job", cascade_delete=True)

class VlogSegment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(foreign_key="vlogjob.id")
    start_second: int
    end_second: int
    voiceover_text: str
    music_vibe: str
    
    # Relationship back to job
    job: VlogJob = Relationship(back_populates="segments")
