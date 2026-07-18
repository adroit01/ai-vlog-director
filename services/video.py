import os
import subprocess
import logging
from typing import List, Tuple
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
from config import settings

logger = logging.getLogger(__name__)

def get_or_create_music(vibe: str, duration: float) -> str:
    """Gets path to music track for the vibe, or generates a synth fallback loop via FFmpeg."""
    import imageio_ffmpeg
    music_dir = settings.STORAGE_DIR / "music"
    music_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize vibe name
    vibe_slug = vibe.lower().replace(" ", "_").replace("&", "and")
    music_path = music_dir / f"{vibe_slug}.mp3"
    
    if music_path.exists() and music_path.stat().st_size > 0:
        return str(music_path)
        
    logger.info(f"Music track not found for '{vibe}'. Generating programmatic ambient synth track...")
    
    # Map vibes to synth frequencies
    freq1, freq2 = 120, 240
    if "lo-fi" in vibe_slug:
        freq1, freq2 = 90, 180
    elif "synthwave" in vibe_slug:
        freq1, freq2 = 140, 280
    elif "ambient" in vibe_slug:
        freq1, freq2 = 80, 160
        
    # Programmatically generate synth using FFmpeg's sine generator resolved via imageio_ffmpeg
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = (
        f'"{ffmpeg_bin}" -f lavfi -i "sine=frequency={freq1}:duration={duration}" '
        f'-f lavfi -i "sine=frequency={freq2}:duration={duration}" '
        f'-filter_complex "amix=inputs=2,volume=0.08" "{music_path}" -y'
    )
    
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(music_path)
    except Exception as e:
        logger.warning(f"Failed to generate synth music: {str(e)}. Using silent fallback...")
        # Create silent fallback using bundled ffmpeg
        silent_cmd = f'"{ffmpeg_bin}" -f lavfi -i "anullsrc=r=44100:c=2:d={duration}" "{music_path}" -y'
        try:
            subprocess.run(silent_cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e2:
            logger.error(f"Failed to generate silent fallback via FFmpeg: {str(e2)}. Creating pure python WAV silence...")
            # If FFmpeg fails completely, use pure Python wave module to create a silent WAV file (saving it with .mp3 suffix so file paths align)
            import wave
            sample_rate = 44100
            num_samples = int(duration * sample_rate)
            with wave.open(str(music_path), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(b'\x00' * (num_samples * 2))
                
        return str(music_path)

def assemble_final_video(
    video_path: str,
    output_path: str,
    voiceover_paths: List[str],
    segments: List[Tuple[int, int]],
    music_vibe: str
) -> bool:
    """Stitches raw video, voiceovers, and background music together with audio ducking."""
    logger.info(f"Starting media assembly for {video_path}...")
    video = None
    bg_music = None
    final_audio = None
    final_video = None
    
    try:
        video = VideoFileClip(video_path)
        # Check for rotation metadata and correct orientation to maintain aspect ratio
        rotation = getattr(video, "rotation", 0)
        if rotation != 0:
            logger.info(f"Detected rotation metadata: {rotation} degrees. Applying correction...")
            video = video.rotated(-rotation, expand=True)
            
        video_duration = video.duration
        
        # 1. Process original audio (atmospheric rumble)
        audio_clips = []
        if video.audio:
            original_audio = video.audio.with_multiply_volume(0.15)
            audio_clips.append(original_audio)
        
        # 2. Add Voiceover Clips
        for idx, voice_path in enumerate(voiceover_paths):
            if os.path.exists(voice_path):
                start_sec, _ = segments[idx]
                voice_clip = AudioFileClip(voice_path).with_start(start_sec)
                audio_clips.append(voice_clip)
        
        # 3. Add Background Music with ducking
        bg_music_path = get_or_create_music(music_vibe, video_duration)
        bg_music = AudioFileClip(bg_music_path)
        from moviepy import afx
        bg_music = bg_music.with_effects([afx.AudioLoop(duration=video_duration)])
        
        # We perform time-based volume modification (ducking)
        # Drops bg music volume during voiceover segments
        def make_duck_filter(t):
            # If t is scalar (float), evaluate directly.
            # If t is a numpy array, process vector.
            import numpy as np
            if isinstance(t, np.ndarray):
                vol = np.ones_like(t) * 0.15  # Normal music volume (15%)
                for start, end in segments:
                    vol[(t >= start) & (t <= end)] = 0.03  # Ducked volume (3%)
                return vol[:, None]
            else:
                for start, end in segments:
                    if start <= t <= end:
                        return 0.03
                return 0.15
                
        # Apply the ducking filter to background music
        ducked_music = bg_music.transform(lambda gf, t: gf(t) * make_duck_filter(t))
        audio_clips.append(ducked_music)
        
        # 4. Composite and write final output
        final_audio = CompositeAudioClip(audio_clips)
        final_video = video.with_audio(final_audio)
        
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            logger=None
        )
        logger.info(f"Successfully rendered output video at: {output_path}")
        return os.path.exists(output_path)
        
    except Exception as e:
        logger.error(f"Video rendering pipeline crashed: {str(e)}", exc_info=True)
        return False
        
    finally:
        # Explicit clean-up to prevent file locking on Windows/Mac
        try:
            if video: video.close()
            if bg_music: bg_music.close()
            if final_audio: final_audio.close()
            if final_video: final_video.close()
        except:
            pass
