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
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import torch
from typing import List, Dict, Optional
import logging
from pathlib import Path

def setup_logging(log_file: str = "transcription.log") -> None:
    """Configure logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Video/Audio to Text Converter")
    parser.add_argument("--input_file", "-i", default="input.mp4", help="Input file path")
    parser.add_argument("--output_file", "-o", default="transcript.txt", help="Output file path")
    parser.add_argument("--model_size", "-m", default="base", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size")
    parser.add_argument("--language", "-l", default=None, help="Force transcription language")
    parser.add_argument("--chunk_size", "-c", type=int, default=10, 
                       help="Chunk size in minutes")
    parser.add_argument("--num_workers", "-w", type=int, default=2,
                       help="Number of parallel workers")
    parser.add_argument("--include_timestamps", "-t", action="store_true",
                       help="Include timestamps in output")
    return parser.parse_args()

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

def transcribe_chunk_with_retry(
    chunk_file: str,
    model: whisper.Whisper,
    max_retries: int = 3,
    language: Optional[str] = None,
    include_timestamps: bool = False
) -> Dict:
    """Transcribe a single chunk with retry mechanism."""
    for attempt in range(max_retries):
        try:
            options = {"language": language} if language else {}
            result = model.transcribe(chunk_file, **options)
            
            if include_timestamps:
                return {
                    "text": result["text"],
                    "segments": [{"text": s["text"], 
                                "start": s["start"],
                                "end": s["end"]} 
                               for s in result["segments"]]
                }
            return {"text": result["text"]}
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(1)
    return {"text": ""}

def transcribe_audio_chunks(
    chunk_files: List[str],
    model_size: str = "base",
    num_workers: int = 2,
    language: Optional[str] = None,
    include_timestamps: bool = False
) -> str:
    """Transcribe audio chunks using parallel processing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_size).to(device)
    
    full_transcript = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                transcribe_chunk_with_retry,
                chunk_file,
                model,
                language=language,
                include_timestamps=include_timestamps
            )
            for chunk_file in chunk_files
        ]
        
        for i, future in enumerate(tqdm(futures, desc="Transcribing chunks")):
            result = future.result()
            if include_timestamps:
                full_transcript.append(f"\n--- Chunk {i + 1} ---\n")
                for segment in result["segments"]:
                    timestamp = f"[{time.strftime('%H:%M:%S', time.gmtime(segment['start']))} -> " \
                              f"{time.strftime('%H:%M:%S', time.gmtime(segment['end']))}]"
                    full_transcript.append(f"{timestamp} {segment['text']}\n")
            else:
                full_transcript.append(f"\n--- Chunk {i + 1} ---\n{result['text']}\n")
    
    return "".join(full_transcript)

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

def main() -> None:
    """Enhanced main function with new features."""
    args = parse_arguments()
    setup_logging()
    
    # Create output directory if it doesn't exist
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    mp3_file = "converted_audio.mp3"
    try:
        with tqdm(total=5, desc="Overall Progress") as pbar:
            process_input_file(args.input_file, mp3_file)
            pbar.update(1)
            
            chunk_files = split_audio_into_chunks(mp3_file, args.chunk_size)
            pbar.update(1)
            
            transcript = transcribe_audio_chunks(
                chunk_files,
                args.model_size,
                args.num_workers,
                args.language,
                args.include_timestamps
            )
            pbar.update(1)
            
            save_transcript(transcript, args.output_file)
            pbar.update(1)
            
            cleanup_temp_files(chunk_files, mp3_file)
            pbar.update(1)
            
        logging.info(f"Transcription completed successfully: {args.output_file}")
        
    except Exception as e:
        logging.error(f"Process failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
