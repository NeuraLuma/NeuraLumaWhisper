### File for CLI ###
from core.model import WhisperModel
from core.converter import AudioConverter
import os
import shutil
import argparse
import jax.numpy as jnp
from datasets import load_dataset

# ToDo: Refactoring, less clutter and better type hints
# ToDo: Generalize this class so WebUI can use it easily
class CLI:
    def __init__(self, dtype=jnp.float16, batch_size=1, hf_checkpoint='openai/whisper-large-v2', hf_load_dataset_options=None):
        self.audio_converter = AudioConverter()
        self.whisper_model = WhisperModel(dtype=dtype, batch_size=batch_size, checkpoint=hf_checkpoint)
        self.audio_data_entries = None

        if hf_load_dataset_options:
            loaded_dataset = load_dataset(
                path=hf_load_dataset_options["source"], 
                name=hf_load_dataset_options["subset"],
                revision=hf_load_dataset_options["revision"], 
                split=hf_load_dataset_options["split"]
                )
            
            print(loaded_dataset)
            
            self.audio_data_entries = [audio_data_entry[hf_load_dataset_options["audio_column"]] for audio_data_entry in loaded_dataset]
            
    # ToDo: Move to utility class
    def is_directory_or_file(self, path):
        if path is None:
            return None
        elif os.path.isdir(path):
            return "directory"
        elif os.path.isfile(path):
            return "file"

    # ToDo: Move to utility class
    def is_audio_or_video(self, file_path):
        _, file = os.path.split(file_path)
        file_type = file.split(".")[-1]

        if file_type == "mp3":
            return "audio"
        elif file_type == "mp4":
            return "video"
        else:
            return None

    def transcribe(self, source_path, output_path, youtube_urls=[], add_timestamps=False, translate=False, cleanup=True):
        source_path_type = self.is_directory_or_file(source_path)

        if source_path_type is None and len(youtube_urls) <= 0:
            raise Exception("Source path is not a directory or a file.")
        elif source_path_type == 'file':
            source_dir, source_file = os.path.split(source_path)
            self.transcribe_file(source_dir=source_dir, source_file=source_file, output_path=output_path, add_timestamps=add_timestamps, translate=translate)
        elif source_path_type == 'directory':
            source_dir = source_path
            self.transcribe_dir(source_dir=source_dir, output_path=output_path, add_timestamps=add_timestamps, translate=translate)
        
        if len(youtube_urls) > 0:
            self.transcribe_youtube(youtube_urls=youtube_urls, output_path=output_path, add_timestamps=add_timestamps, translate=translate)
        
        if self.audio_data_entries and len(self.audio_data_entries) > 0:
            self.transcribe_raw_audio(data_entries=self.audio_data_entries, output_path=output_path, add_timestamps=add_timestamps, translate=translate)
        
        # ToDo: Make this conditional, perhaps but this in another command
        if cleanup:
            if os.path.exists('temp'):
                shutil.rmtree('temp')
    
    def transcribe_raw_audio(self, data_entries, output_path, add_timestamps=False, translate=False):
        transcriptions = []
        for idx, data_entry in enumerate(data_entries):
            if not translate:
                print(f"Transcribing audio file {idx+1} of {len(data_entries)}")
                transcription = self.whisper_model.transcribe(inputs=data_entry, add_timestamps=add_timestamps)
                print("Done!")
            else:
                print(f"Translating audio file {idx+1} of {len(data_entries)}")
                transcription = self.whisper_model.translate(inputs=data_entry, add_timestamps=add_timestamps)
                print("Done!")
            transcriptions.append(transcription)
        
        # ToDo: This code is duplicated, it should be refactored in a way that can be reused and allows hf_datasets as well
        for idx, transcription in enumerate(transcriptions):
            base_output_filename = str(idx)
            if not translate:
                base_output_filename += ".transcription"
            else:
                base_output_filename += ".translation"
            
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            with open(os.path.join(output_path, base_output_filename + ".txt"), 'w') as f:
                f.write(transcription['text'])
            
            if add_timestamps:
                with open(os.path.join(output_path, base_output_filename + ".sbv"), "w", encoding="utf-8") as sbv_file:
                        # Iterate through each chunk in the transcription
                        for chunk in transcription['chunks']:
                            # Extract the timestamp and text from the chunk
                            start_time, end_time = chunk['timestamp']
                            text = chunk['text']
                            
                            # Convert the timestamps to the SBV format (hh:mm:ss.mmm)
                            start_minutes, start_seconds = divmod(start_time, 60)
                            start_hours, start_minutes = divmod(start_minutes, 60)
                            end_minutes, end_seconds = divmod(end_time, 60)
                            end_hours, end_minutes = divmod(end_minutes, 60)
                            
                            start_time_sbv = f"{int(start_hours):02d}:{int(start_minutes):02d}:{start_seconds:06.3f}"
                            end_time_sbv = f"{int(end_hours):02d}:{int(end_minutes):02d}:{end_seconds:06.3f}"
                            
                            # Write the timestamp and text to the file
                            sbv_file.write(f"{start_time_sbv},{end_time_sbv}\n{text}\n\n")
            
            

    def transcribe_youtube(self, youtube_urls, output_path, add_timestamps=False, translate=False):
        converted_paths, _ = self.audio_converter.convert_multiple_from_youtube(urls=youtube_urls)

        for path in converted_paths:
            self.transcribe_file(source_dir=os.path.dirname(path), source_file=os.path.basename(path), output_path=output_path, add_timestamps=add_timestamps, translate=translate)

    def transcribe_file(self, source_dir, source_file, output_path, add_timestamps=False, translate=False):
        file_path = os.path.join(source_dir, source_file)

        # Transcription logic for a single file
        file_type = self.is_audio_or_video(file_path)
        if file_type == 'audio':
            source_audio_file_path = file_path
        elif file_type == 'video':
            source_audio_file_path = self.audio_converter.convert_from_video(input_path=source_dir, input_file_name=source_file, output_path=output_path)
        else:
            raise Exception("Unknown file type provided.")
        
        transcription = {}

        if not translate:
            transcription = self.whisper_model.transcribe(inputs=source_audio_file_path, add_timestamps=add_timestamps)
        else:
            transcription = self.whisper_model.translate(inputs=source_audio_file_path, add_timestamps=add_timestamps)
        
        print(transcription)

        base_output_filename =  os.path.basename(file_path).split(".")[0]

        if not translate:
            base_output_filename += ".transcription"
        else:
            base_output_filename += ".translation"
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        with open(os.path.join(output_path, base_output_filename + ".txt"), 'w') as f:
            f.write(transcription['text'])
        
        if add_timestamps:
            with open(os.path.join(output_path, base_output_filename + ".sbv"), "w", encoding="utf-8") as sbv_file:
                    # Iterate through each chunk in the transcription
                    for chunk in transcription['chunks']:
                        # Extract the timestamp and text from the chunk
                        start_time, end_time = chunk['timestamp']
                        text = chunk['text']
                        
                        # Convert the timestamps to the SBV format (hh:mm:ss.mmm)
                        start_minutes, start_seconds = divmod(start_time, 60)
                        start_hours, start_minutes = divmod(start_minutes, 60)
                        end_minutes, end_seconds = divmod(end_time, 60)
                        end_hours, end_minutes = divmod(end_minutes, 60)
                        
                        start_time_sbv = f"{int(start_hours):02d}:{int(start_minutes):02d}:{start_seconds:06.3f}"
                        end_time_sbv = f"{int(end_hours):02d}:{int(end_minutes):02d}:{end_seconds:06.3f}"
                        
                        # Write the timestamp and text to the file
                        sbv_file.write(f"{start_time_sbv},{end_time_sbv}\n{text}\n\n")


    def transcribe_dir(self, source_dir, output_path, add_timestamps=False, translate=False):
        # ToDo: Cleanup temp files (conversions, downloads)

        files = self.collect_files_recursively(source_dir)

        print(files)

        audio_files = []
        video_files = []
        transcriptions = {}

        for file in files:
            file_type = self.is_audio_or_video(file)
            if file_type == 'audio':
                audio_files.append(file)
            elif file_type == 'video':
                video_files.append(file)

        for audio_file in audio_files:
            if not translate:
                transcriptions[audio_file] = self.whisper_model.transcribe(audio_file, add_timestamps=add_timestamps)
            else:
                transcriptions[audio_file] = self.whisper_model.translate(audio_file, add_timestamps=add_timestamps)
        
        if len(video_files) > 0:
            # Files with the same name in nested directory could cause an error
            output_file_names = [os.path.basename(video_file_path).split(".")[0] + ".mp3" for video_file_path in video_files]
            video_audio_files = self.audio_converter.convert_multiple_from_videos(input_source_paths=video_files, output_file_names=output_file_names)

            for video_audio_file in video_audio_files:
                if not translate:
                    transcriptions[video_audio_file] = self.whisper_model.transcribe(video_audio_file, add_timestamps=add_timestamps)
                else:
                    transcriptions[video_audio_file] = self.whisper_model.translate(video_audio_file, add_timestamps=add_timestamps)
        
        print(transcriptions)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for file_path, transcription_content in transcriptions.items():
             # Extract the base filename from the path
            base_filename = os.path.basename(file_path).split(".")[0]

            print(transcription_content)
            # Get file name
            if not translate:
                base_filename = base_filename + ".transcription"
            else:
                base_filename = base_filename + ".translation"
            # Save to output directory
            with open(os.path.join(output_path, base_filename + ".txt"), "w") as f:
                f.write(transcription_content['text'])
            
            if add_timestamps:
                with open(os.path.join(output_path, base_filename + ".sbv"), "w", encoding="utf-8") as sbv_file:
                    # Iterate through each chunk in the transcription
                    for chunk in transcription_content['chunks']:
                        # Extract the timestamp and text from the chunk
                        start_time, end_time = chunk['timestamp']
                        text = chunk['text']
                        
                        # Convert the timestamps to the SBV format (hh:mm:ss.mmm)
                        start_minutes, start_seconds = divmod(start_time, 60)
                        start_hours, start_minutes = divmod(start_minutes, 60)
                        end_minutes, end_seconds = divmod(end_time, 60)
                        end_hours, end_minutes = divmod(end_minutes, 60)
                        
                        start_time_sbv = f"{int(start_hours):02d}:{int(start_minutes):02d}:{start_seconds:06.3f}"
                        end_time_sbv = f"{int(end_hours):02d}:{int(end_minutes):02d}:{end_seconds:06.3f}"
                        
                        # Write the timestamp and text to the file
                        sbv_file.write(f"{start_time_sbv},{end_time_sbv}\n{text}\n\n")

    def collect_files_recursively(self, source_dir):
        file_paths = []

        for path in os.listdir(source_dir):
            print(path)
            if os.path.isdir(os.path.join(source_dir, path)):
                print('isdir')
                file_paths += self.collect_files_recursively(os.path.join(source_dir, path))
            elif os.path.isfile(os.path.join(source_dir, path)):
                print ('isfile')
                file_paths.append(os.path.join(source_dir, path))
        
        print('return', file_paths)

        return file_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuraLumaWhisper transcription tool")
    parser.add_argument('-s', '--source', help='Path to the file(s) to be transcribed', required=False)
    parser.add_argument('-y', '--youtube', help="URL(s) of the YouTube video(s) to be transcribed, seperate with semicolon (';')", required=False)
    parser.add_argument('-ld', '--hf_load_dataset', help='HF dataset to load', required=False)
    parser.add_argument('-ldc', '--hf_load_dataset_column', help='HF dataset column to load', default='audio')
    parser.add_argument('-ldr', '--hf_load_dataset_revision', help='HF dataset revision to load', default='main')
    parser.add_argument('-ldst', '--hf_load_dataset_subset', help='HF dataset subset to load', required=False)
    parser.add_argument('-ldsp', '--hf_load_dataset_split', help='HF dataset split to load', required=False)
    parser.add_argument('-o', '--output', help='Path to the directory for the transcribed files', required=True)
    parser.add_argument('-ts', '--timestamp', help='Activates timestamps. Adds a seperate file with sbv extension', action='store_true')
    parser.add_argument('-tl', '--translate', help='Sets the mode to translation', action='store_true')
    parser.add_argument('-d', '--dtype', help='Sets the dtype to use', default='float16', choices=['float16', 'bfloat16', 'float32', 'float64'])
    parser.add_argument('-b', '--batch_size', help='Sets the batch size for inference', default=1, type=int)
    parser.add_argument('-hfc', '--hf_checkpoint', help='Sets the hf checkpoint to use', default='openai/whisper-large-v2')
    # ToDo: Add Option to create HuggingFace Dataset
    # ToDo: Add Option to push to HuggingFace Hub
    # ToDo: Do some temp post-cleanup
    # ToDo: Add name scheming option (single file = name_scheme, multiple files = name_scheme_idx, name_scheme => base_filename)
    # ToDo: Add more verbose logging / progress
    
    args = parser.parse_args()
    source_path = args.source
    output_path = args.output
    youtube = args.youtube
    add_timestamps = args.timestamp
    translate = args.translate
    dtype = args.dtype
    batch_size = args.batch_size
    hf_checkpoint = args.hf_checkpoint
    hf_load_dataset = args.hf_load_dataset
    hf_load_dataset_column = args.hf_load_dataset_column
    hf_load_dataset_revision = args.hf_load_dataset_revision
    hf_load_dataset_subset = args.hf_load_dataset_subset
    hf_load_dataset_split = args.hf_load_dataset_split

    if not youtube and not source_path and not hf_load_dataset:
        raise Exception("Please specify either -y --youtube and/or -s --source and/or -ld --hf_load_dataset")
    
    if youtube is None:
        youtube = ""
    
    hf_load_dataset_options = None
    if hf_load_dataset:
        hf_load_dataset_options = {
            "source": hf_load_dataset, 
            "audio_column": hf_load_dataset_column, 
            "revision": hf_load_dataset_revision,
            "subset": hf_load_dataset_subset,
            "split": hf_load_dataset_split
            }
    
    print(hf_load_dataset_options)

    cli = CLI(dtype=getattr(jnp, dtype), batch_size=batch_size, hf_checkpoint=hf_checkpoint, hf_load_dataset_options=hf_load_dataset_options)

    youtube_urls = youtube.split(";")

    cli.transcribe(source_path=source_path, output_path=output_path, youtube_urls=youtube_urls, add_timestamps=add_timestamps, translate=translate)