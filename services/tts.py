import os
import asyncio
import httpx
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from config import settings

logger = logging.getLogger(__name__)

class TTSAdapter(ABC):
    @abstractmethod
    async def generate_audio(self, text: str, output_path: str, voice_id: str = None) -> bool:
        """Generates audio file from text. Returns True on success, False on failure."""
        pass

class MockTTSAdapter(TTSAdapter):
    async def generate_audio(self, text: str, output_path: str, voice_id: str = None) -> bool:
        logger.info(f"[MockTTS] Synthesizing: '{text[:40]}...' -> {output_path}")
        # Create a tiny mock MP3 file
        with open(output_path, "wb") as f:
            f.write(b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 200)
        return os.path.exists(output_path)

class EdgeTTSAdapter(TTSAdapter):
    async def generate_audio(self, text: str, output_path: str, voice_id: str = None) -> bool:
        import edge_tts
        # Default voice is en-US-GuyNeural
        voice = voice_id or "en-US-GuyNeural"
        logger.info(f"[EdgeTTS] Synthesizing with voice {voice}: '{text[:40]}...' -> {output_path}")
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
            return os.path.exists(output_path) and os.path.getsize(output_path) > 0
        except Exception as e:
            logger.error(f"[EdgeTTS] Error during voice synthesis: {str(e)}", exc_info=True)
            return False

class ElevenLabsAdapter(TTSAdapter):
    async def generate_audio(self, text: str, output_path: str, voice_id: str = None) -> bool:
        voice = voice_id or "21m00Tcm4TlvDq8ikWAM" # Default Rachel voice
        api_key = settings.ELEVENLABS_API_KEY
        if not api_key:
            logger.error("[ElevenLabs] Error: ELEVENLABS_API_KEY is not configured.")
            return False
            
        logger.info(f"[ElevenLabs] Synthesizing: '{text[:40]}...' -> {output_path}")
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=data, headers=headers, timeout=60.0)
                if response.status_code == 200:
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    return True
                else:
                    logger.error(f"[ElevenLabs] HTTP Error {response.status_code}: {response.text}")
                    return False
            except Exception as e:
                logger.error(f"[ElevenLabs] Exception during API call: {str(e)}", exc_info=True)
                return False

class F5TTSAdapter(TTSAdapter):
    async def generate_audio(self, text: str, output_path: str, voice_id: str = None) -> bool:
        host = getattr(settings, "F5_TTS_HOST", "http://localhost:9000")
        
        # Try both the FastAPI endpoint (/tts/) and the OpenAI compatible endpoint (/v1/audio/speech)
        async with httpx.AsyncClient() as client:
            # 1. Try FastAPI /tts/ endpoint first
            try:
                url = f"{host}/tts/"
                payload = {
                    "gen_text": text,
                    "ref_audio": voice_id or "",
                    "speed": 1.0,
                    "remove_silence": True
                }
                logger.info(f"[F5-TTS] Attempting synthesis via FastAPI endpoint: {url}")
                response = await client.post(url, json=payload, timeout=600.0)
                if response.status_code == 200:
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    return True
                else:
                    logger.error(f"[F5-TTS] FastAPI endpoint returned {response.status_code}: {response.text}")
            except Exception as e:
                logger.warning(f"[F5-TTS] FastAPI endpoint failed with exception: {repr(e)}. Trying OpenAI compatible endpoint...")
                
            # 2. Try OpenAI compatible endpoint (/v1/audio/speech)
            try:
                url = f"{host}/v1/audio/speech"
                payload = {
                    "model": "f5-tts",
                    "input": text,
                    "voice": voice_id or "default"
                }
                logger.info(f"[F5-TTS] Attempting synthesis via OpenAI endpoint: {url}")
                response = await client.post(url, json=payload, timeout=600.0)
                if response.status_code == 200:
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    return True
                else:
                    logger.error(f"[F5-TTS] OpenAI endpoint returned {response.status_code}: {response.text}")
            except Exception as e:
                logger.error(f"[F5-TTS] OpenAI endpoint failed with exception: {repr(e)}")
                
            return False

class BarkAdapter(TTSAdapter):
    def __init__(self):
        self.initialized = False
        self.processor = None
        self.model = None

    def _lazy_init(self):
        if self.initialized:
            return
        # Lazy load to speed up startup time
        from transformers import AutoProcessor, BarkModel
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[Bark] Loading models onto {device}...")
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = BarkModel.from_pretrained("suno/bark-small").to(device)
        self.device = device
        self.initialized = True

    async def generate_audio(self, text: str, output_path: str, voice_id: str = None) -> bool:
        logger.info(f"[Bark] Synthesizing: '{text[:40]}...' -> {output_path}")
        try:
            # Run blocking model inference in a separate thread pool executor
            return await asyncio.to_thread(self._generate_audio_sync, text, output_path, voice_id)
        except Exception as e:
            logger.error(f"[Bark] Exception: {str(e)}", exc_info=True)
            return False

    def _generate_audio_sync(self, text: str, output_path: str, voice_id: str = None) -> bool:
        self._lazy_init()
        import scipy.io.wavfile
        import torch
        
        # Bark voice preset
        preset = voice_id or "v2/en_speaker_6"
        
        inputs = self.processor(text, voice_preset=preset).to(self.device)
        with torch.no_grad():
            audio_array = self.model.generate(**inputs)
            audio_array = audio_array.cpu().numpy().squeeze()
            
        sample_rate = self.model.generation_config.sample_rate
        
        # Save temporary wav
        temp_wav = output_path + ".temp.wav"
        scipy.io.wavfile.write(temp_wav, sample_rate, audio_array)
        
        # Convert to mp3 using ffmpeg to standardise formats
        import subprocess
        subprocess.run(
            ["ffmpeg", "-i", temp_wav, "-codec:a", "libmp3lame", "-qscale:a", "2", output_path, "-y"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
            
        return os.path.exists(output_path)

def get_tts_adapter(provider: str) -> TTSAdapter:
    provider = provider.lower()
    if provider == "elevenlabs":
        return ElevenLabsAdapter()
    elif provider == "bark":
        return BarkAdapter()
    elif provider == "edge-tts":
        return EdgeTTSAdapter()
    elif provider in ("f5-tts", "f5tts"):
        return F5TTSAdapter()
    else:
        return MockTTSAdapter()
