from core.model import WhisperModel
from core.converter import AudioConverter
import os
import shutil
import jax.numpy as jnp
from datasets import load_dataset, Dataset, Audio

# ToDo: Refactoring, less clutter and better type hints
# ToDo: Make return values also be returnable directly so it can be used in workflows
# ToDo: Accept other audio and video formats
class NeuraLumaWhisperPipeline:
    def __init__(self, dtype=jnp.float16, batch_size=1, hf_checkpoint='openai/whisper-large-v2', hf_load_dataset_options=None):
        self.audio_converter = AudioConverter()
        self.whisper_model = WhisperModel(dtype=dtype, batch_size=batch_size, checkpoint=hf_checkpoint)
        self.audio_data_entries = None

        # ToDo: Check if subsets work properly
        # ToDo: Check whether local datasets work as well
        if hf_load_dataset_options:
            loaded_dataset = load_dataset(
                path=hf_load_dataset_options["source"], 
                name=hf_load_dataset_options["subset"],
                revision=hf_load_dataset_options["revision"], 
                split=hf_load_dataset_options["split"]
                )
            
            print(loaded_dataset)
            
            # ToDo: If an error occurs here it might be due to: wrong column name, wrong split. wrong subset. add warning here
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

    def transcribe(self, source_path, output_path=None, hf_save_dataset_options=None, youtube_urls=[], add_timestamps=False, translate=False, cleanup=True):
        source_path_type = self.is_directory_or_file(source_path)

        if source_path_type is None and len(youtube_urls) <= 0:
            raise Exception("Source path is not a directory or a file.")
        elif source_path_type == 'file':
            source_dir, source_file = os.path.split(source_path)
            self.transcribe_file(source_dir=source_dir, source_file=source_file, output_path=output_path, hf_save_dataset_options=hf_save_dataset_options, add_timestamps=add_timestamps, translate=translate)
        elif source_path_type == 'directory':
            source_dir = source_path
            self.transcribe_dir(source_dir=source_dir, output_path=output_path, hf_save_dataset_options=hf_save_dataset_options, add_timestamps=add_timestamps, translate=translate)
        
        if len(youtube_urls) > 0:
            self.transcribe_youtube(youtube_urls=youtube_urls, output_path=output_path, hf_save_dataset_options=hf_save_dataset_options, add_timestamps=add_timestamps, translate=translate)
        
        if self.audio_data_entries and len(self.audio_data_entries) > 0:
            self.transcribe_raw_audio(data_entries=self.audio_data_entries, output_path=output_path, hf_save_dataset_options=hf_save_dataset_options, add_timestamps=add_timestamps, translate=translate)
        
        # ToDo: Make this conditional, perhaps but this in another command
        if cleanup:
            print("Cleaning up and removing temp folder")
            if os.path.exists('temp'):
                shutil.rmtree('temp')
            print("Done!")
    def get_timestamped_sbv_text(self, transcription):
        output = []
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
            output.append((f"{start_time_sbv},{end_time_sbv}\n{text}"))
        
        return "\n\n".join(output)
    
    def write_transcription_to_file(self, transcription, base_ouput_filename, output_path, translate=False, add_timestamps=False):
        if not translate:
            base_output_filename = base_ouput_filename + ".transcription"
        else:
            base_output_filename = base_ouput_filename + ".translation"
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        with open(os.path.join(output_path, base_output_filename + ".txt"), 'w') as f:
            f.write(transcription['text'])
        
        if add_timestamps:
            with open(os.path.join(output_path, base_output_filename + ".sbv"), "w", encoding="utf-8") as sbv_file:
                sbv_text = self.get_timestamped_sbv_text(transcription)
                # Write the timestamp and text to the file
                sbv_file.write(sbv_text)
    
    def write_transcriptions_to_hf_dataset(self, audio_source_entries, transcriptions, hf_save_dataset_options, add_timestamps=False, translate=False):
        if not transcriptions or len(transcriptions) <= 0:
            return
        
        # Build a dataset from audio source entries and transcriptions
        # Check if audio_source_entries is a list of strings or is list of dicts (probably a hf dataset)

        # We already have a list of paths most likely
        audio_column_name = hf_save_dataset_options["audio_column"]
        text_column_name = hf_save_dataset_options["text_column"]
        sbv_column_name = hf_save_dataset_options["text_sbv_column"]
        #subset = hf_save_dataset_options["subset"]
    
        if isinstance(audio_source_entries, list) and all(isinstance(item, str) for item in audio_source_entries):
            hf_dataset = Dataset.from_dict({
                audio_column_name: audio_source_entries,
                text_column_name: [transcription["text"] for transcription in transcriptions],
                sbv_column_name: [self.get_timestamped_sbv_text(transcription) for transcription in transcriptions],
            }).cast_column(audio_column_name, Audio())
        elif isinstance(audio_source_entries, list) and all(isinstance(item, dict) for item in audio_source_entries):
            # ToDo: Audio Columns are still somehow in the wrong format, as it appears in the wrong format in the parquet dataset viewer
            hf_dataset = Dataset.from_dict({
                audio_column_name: audio_source_entries,
                text_column_name: [transcription["text"] for transcription in transcriptions],
                sbv_column_name: [self.get_timestamped_sbv_text(transcription) for transcription in transcriptions],
            })
        else:
            raise Exception("The provided type for audio_columns isn't supported")

        split = hf_save_dataset_options["split"]
        target_repository = hf_save_dataset_options["target"]
        revision = hf_save_dataset_options["revision"]
        private = hf_save_dataset_options["private"]

        # ToDo: Make this conditional, return value might be desired to be the dataset itself or save to file
        hf_dataset.push_to_hub(target_repository, split=split, branch=revision, private=private)

    def transcribe_raw_audio(self, data_entries, output_path, hf_save_dataset_options, add_timestamps=False, translate=False):
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
        if output_path:
            for idx, transcription in enumerate(transcriptions):
                base_output_filename = str(idx)
                self.write_transcription_to_file(transcription=transcription, base_ouput_filename=base_output_filename, output_path=output_path, translate=translate, add_timestamps=add_timestamps)
                
        if hf_save_dataset_options:
            self.write_transcriptions_to_hf_dataset(audio_source_entries=data_entries, transcriptions=transcriptions, hf_save_dataset_options=hf_save_dataset_options, add_timestamps=add_timestamps, translate=translate)

    def transcribe_youtube(self, youtube_urls, output_path, hf_save_dataset_options, add_timestamps=False, translate=False):
        converted_paths, _ = self.audio_converter.convert_multiple_from_youtube(urls=youtube_urls)

        for path in converted_paths:
            print(f"Transcribing audio file sourced from YouTube {path}")
            self.transcribe_file(source_dir=os.path.dirname(path), source_file=os.path.basename(path), output_path=output_path, hf_save_dataset_options=hf_save_dataset_options, add_timestamps=add_timestamps, translate=translate)

    def transcribe_file(self, source_dir, source_file, output_path, hf_save_dataset_options, add_timestamps=False, translate=False):
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
            print(f"Transcribing {file_type} file {source_file}")
            transcription = self.whisper_model.transcribe(inputs=source_audio_file_path, add_timestamps=add_timestamps)
            print("Done!")
        else:
            print(f"Translating {file_type} file {source_file}")
            transcription = self.whisper_model.translate(inputs=source_audio_file_path, add_timestamps=add_timestamps)
            print("Done!")
        
        print(transcription)

        base_output_filename = os.path.basename(file_path).split(".")[0]

        if output_path:
            self.write_transcription_to_file(transcription=transcription, base_ouput_filename=base_output_filename, output_path=output_path, translate=translate, add_timestamps=add_timestamps)
        
        if hf_save_dataset_options:
            self.write_transcriptions_to_hf_dataset(audio_source_entries=[source_audio_file_path], transcriptions=[transcription], hf_save_dataset_options=hf_save_dataset_options, add_timestamps=add_timestamps, translate=translate)


    def transcribe_dir(self, source_dir, output_path, hf_save_dataset_options, add_timestamps=False, translate=False):
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
                    print(f"Transcribing {file_type} file {video_audio_file}")
                    transcriptions[video_audio_file] = self.whisper_model.transcribe(video_audio_file, add_timestamps=add_timestamps)
                    print("Done!")
                else:
                    print(f"Translating {file_type} file {video_audio_file}")
                    transcriptions[video_audio_file] = self.whisper_model.translate(video_audio_file, add_timestamps=add_timestamps)
                    print("Done!")
        
        print(transcriptions)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if output_path:
            for file_path, transcription_content in transcriptions.items():
                # Extract the base filename from the path
                base_filename = os.path.basename(file_path).split(".")[0]

                print(transcription_content)
                self.write_transcription_to_file(transcription=transcription_content, base_ouput_filename=base_filename, output_path=output_path, translate=translate, add_timestamps=add_timestamps)
        
        if hf_save_dataset_options:
            self.write_transcriptions_to_hf_dataset(audio_source_entries=audio_files, transcriptions=transcriptions, hf_save_dataset_options=hf_save_dataset_options, add_timestamps=add_timestamps, translate=translate)

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
