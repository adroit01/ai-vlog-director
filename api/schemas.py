from pydantic import BaseModel, Field, computed_field
from typing import List, Optional
from datetime import datetime

class ProcessVlogRequest(BaseModel):
    video_id: str = Field(..., description="ID of the uploaded raw video.")
    prompt_hint: Optional[str] = Field(None, description="Optional description of the video or historical hint.")
    music_vibe: str = Field("Cinematic & Adventurous", description="Background music genre or style.")
    tts_provider: str = Field("edge-tts", description="TTS Provider to use (edge-tts, elevenlabs, bark, mock).")
    voice_id: Optional[str] = Field(None, description="ID of voice preset to use.")
    model_name: str = Field("gemini-2.5-flash", description="AI Model to run the pipeline.")

class VlogSegmentResponse(BaseModel):
    start_second: int
    end_second: int
    voiceover_text: str
    music_vibe: str

    class Config:
        from_attributes = True

class VlogJobResponse(BaseModel):
    id: str
    status: str
    progress_percentage: int
    error_message: Optional[str] = None
    video_id: str
    filename: str
    prompt_hint: Optional[str] = None
    music_vibe: str
    tts_provider: str
    voice_id: Optional[str] = None
    model_name: str
    created_at: datetime
    updated_at: datetime
    segments: List[VlogSegmentResponse] = []
    output_video_path: Optional[str] = None

    @computed_field
    @property
    def job_id(self) -> str:
        return self.id

    class Config:
        from_attributes = True
