# Speech to Text Converter for Video and Audio Files

This project converts video or audio files to text using the Whisper model. It processes the audio (either extracted from video or directly from audio files), splits it into chunks, transcribes the chunks, and saves the transcript to a text file.

## Requirements

Install the required packages using pip:

```bash
pip install moviepy openai-whisper ffmpeg-python setuptools-rust pydub
```

## Usage

1. Place your video or audio file in the project directory.
2. Run the script with the default parameters:

```bash
python video_to_text.py
```

3. The transcript will be saved to `full_transcript.txt` by default.

## Functions

- `convert_video_to_mp3(video_path: str, output_path: str) -> None`: Converts a video file to MP3 format.
- `split_audio_into_chunks(mp3_path: str, chunk_length_minutes: int = 10) -> List[str]`: Splits an MP3 file into smaller chunks.
- `transcribe_audio_chunks(chunk_files: List[str], model_size: str = "base") -> str`: Transcribes audio chunks using the Whisper model.
- `save_transcript(transcript: str, output_file: str) -> None`: Saves the transcript to a file.
- `cleanup_temp_files(chunk_files: List[str], mp3_file: str) -> None`: Removes temporary files.

## Example

To process a video file named `input.mp4` and save the transcript to `full_transcript.txt`, run:

```bash
python video_to_text.py
```

You can also specify different input and output files:

```bash
python video_to_text.py --video_file your_video.mp4 --output_file your_transcript.txt
```

## License

This project is licensed under the MIT License.
