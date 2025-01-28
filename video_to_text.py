"""
Speech to Text converter for video and audio files
Requirements:
pip install moviepy openai-whisper ffmpeg-python setuptools-rust pydub
"""

from typing import List
import os
from moviepy.editor import AudioFileClip
import whisper
from pydub import AudioSegment

def convert_video_to_mp3(video_path: str, output_path: str) -> None:
    """Convert video file to MP3 format."""
    try:
        print("Converting video to MP3...")
        audio_clip = AudioFileClip(video_path)
        audio_clip.write_audiofile(output_path)
        audio_clip.close()
        print(f"Converted to MP3: {output_path}")
    except Exception as e:
        print(f"Error converting video to MP3: {str(e)}")
        raise

def split_audio_into_chunks(
    mp3_path: str, 
    chunk_length_minutes: int = 10
) -> List[str]:
    """Split MP3 file into smaller chunks."""
    try:
        print("Splitting MP3 file into chunks...")
        audio = AudioSegment.from_file(mp3_path)
        chunk_length_ms = chunk_length_minutes * 60 * 1000
        chunks = [audio[i:i + chunk_length_ms] 
                 for i in range(0, len(audio), chunk_length_ms)]

        chunk_files = []
        for i, chunk in enumerate(chunks):
            chunk_file = f"chunk_{i + 1}.mp3"
            chunk.export(chunk_file, format="mp3")
            chunk_files.append(chunk_file)
            print(f"Created chunk: {chunk_file}")
        return chunk_files
    except Exception as e:
        print(f"Error splitting audio: {str(e)}")
        raise

def transcribe_audio_chunks(
    chunk_files: List[str], 
    model_size: str = "base"
) -> str:
    """Transcribe audio chunks using Whisper model."""
    try:
        print(f"Loading Whisper model: {model_size}")
        model = whisper.load_model(model_size)
        full_transcript = ""

        for i, chunk_file in enumerate(chunk_files):
            print(f"Processing chunk: {chunk_file}")
            result = model.transcribe(chunk_file)
            full_transcript += f"--- Chunk {i + 1} ---\n"
            full_transcript += result['text'] + "\n"

        return full_transcript
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        raise

def save_transcript(transcript: str, output_file: str) -> None:
    """Save transcript to a file."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"Transcript saved to: {output_file}")
    except Exception as e:
        print(f"Error saving transcript: {str(e)}")
        raise

def cleanup_temp_files(chunk_files: List[str], mp3_file: str) -> None:
    """Remove temporary files."""
    try:
        for file in chunk_files:
            os.remove(file)
        os.remove(mp3_file)
        print("Temporary files cleaned up")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

def is_audio_file(file_path: str) -> bool:
    """Check if the file is an audio file based on extension."""
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'}
    return os.path.splitext(file_path)[1].lower() in audio_extensions

def process_input_file(input_file: str, mp3_file: str) -> None:
    """Process input file based on its type."""
    if is_audio_file(input_file):
        if input_file.lower().endswith('.mp3'):
            # If input is already MP3, just copy it
            import shutil
            shutil.copy2(input_file, mp3_file)
            print(f"Using audio file directly: {input_file}")
        else:
            # Convert other audio formats to MP3
            audio = AudioSegment.from_file(input_file)
            audio.export(mp3_file, format="mp3")
            print(f"Converted audio to MP3: {mp3_file}")
    else:
        convert_video_to_mp3(input_file, mp3_file)

def main(
    input_file: str = "input.mp4",
    output_file: str = "full_transcript.txt",
    model_size: str = "base"
) -> None:
    """Main function to process video/audio and generate transcript."""
    mp3_file = "converted_audio.mp3"
    
    try:
        process_input_file(input_file, mp3_file)
        chunk_files = split_audio_into_chunks(mp3_file)
        transcript = transcribe_audio_chunks(chunk_files, model_size)
        save_transcript(transcript, output_file)
        cleanup_temp_files(chunk_files, mp3_file)
    except Exception as e:
        print(f"Process failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
