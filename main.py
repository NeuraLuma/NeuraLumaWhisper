### File for CLI ###
import argparse
import jax.numpy as jnp
from core.pipeline import NeuraLumaWhisperPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuraLumaWhisper transcription tool")
    parser.add_argument('-s', '--source', help='Path to the file(s) to be transcribed', required=False)
    parser.add_argument('-y', '--youtube', help="URL(s) of the YouTube video(s) to be transcribed, seperate with semicolon (';')", required=False)
    parser.add_argument('-ld', '--hf_load_dataset', help='HF dataset to load', required=False)
    parser.add_argument('-ldc', '--hf_load_dataset_column', help='HF dataset column to load', default='audio')
    parser.add_argument('-ldr', '--hf_load_dataset_revision', help='HF dataset revision to load', default='main')
    parser.add_argument('-ldst', '--hf_load_dataset_subset', help='HF dataset subset to load', required=False)
    parser.add_argument('-ldsp', '--hf_load_dataset_split', help='HF dataset split to load', required=False)
    parser.add_argument('-o', '--output', help='Path to the directory for the transcribed files', required=False)
    parser.add_argument('-sd', '--hf_save_dataset', help='HF dataset Hub Id to save to e.g. my_user/my_dataset', required=False)
    parser.add_argument('-sdp', '--hf_save_dataset_private', help='Set whether the HF dataset is private', default=True, choices=[True, False], type=bool)
    parser.add_argument('-sdca', '--hf_save_dataset_column_audio', help='HF dataset column to save audio to', default='audio')
    parser.add_argument('-sdct', '--hf_save_dataset_column_text', help='HF dataset column to save text to', default='text')
    parser.add_argument('-sdcsbv', '--hf_save_dataset_column_text_sbv', help='HF dataset column to save text with (SBV formatted) timestamps to', default='sbv')
    parser.add_argument('-sdr', '--hf_save_dataset_revision', help='HF dataset revision to save to', default='main')
    #parser.add_argument('-sds', '--hf_save_dataset_subset', help='HF dataset subset to save to', required=False)
    parser.add_argument('-sdsp', '--hf_save_dataset_split', help='HF dataset split to save to', required=False)
    parser.add_argument('-ts', '--timestamp', help='Activates timestamps. Adds a seperate file with sbv extension', action='store_true')
    parser.add_argument('-tl', '--translate', help='Sets the mode to translation', action='store_true')
    parser.add_argument('-d', '--dtype', help='Sets the dtype to use', default='float16', choices=['float16', 'bfloat16', 'float32', 'float64'])
    parser.add_argument('-b', '--batch_size', help='Sets the batch size for inference', default=1, type=int)
    parser.add_argument('-hfc', '--hf_checkpoint', help='Sets the hf checkpoint to use', default='openai/whisper-large-v2')
    # ToDo: Do some temp post-cleanup
    # ToDo: Add name scheming option (single file = name_scheme, multiple files = name_scheme_idx, name_scheme => base_filename)
    # ToDo: Add option to save huggingface dataset to disk and optionally push to hub
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
    # Loading Dataset Options
    hf_load_dataset = args.hf_load_dataset
    hf_load_dataset_column = args.hf_load_dataset_column
    hf_load_dataset_revision = args.hf_load_dataset_revision
    hf_load_dataset_subset = args.hf_load_dataset_subset
    hf_load_dataset_split = args.hf_load_dataset_split
    # Saving Dataset Options
    hf_save_dataset = args.hf_save_dataset
    hf_save_dataset_private = args.hf_save_dataset_private
    hf_save_dataset_column_audio = args.hf_save_dataset_column_audio
    hf_save_dataset_column_text = args.hf_save_dataset_column_text
    hf_save_dataset_column_text_sbv = args.hf_save_dataset_column_text_sbv
    hf_save_dataset_revision = args.hf_save_dataset_revision
    #hf_save_dataset_subset = args.hf_save_dataset_subset
    hf_save_dataset_split = args.hf_save_dataset_split

    # ToDo: Implement subsets

    if not youtube and not source_path and not hf_load_dataset:
        raise Exception("Please specify either -y --youtube and/or -s --source and/or -ld --hf_load_dataset")
    
    if not output_path and not hf_save_dataset:
        raise Exception("Please specify either -o --output and/or -sd --hf_save_dataset")
    
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
    
    hf_save_dataset_options = None
    if hf_save_dataset:
        hf_save_dataset_options = {
            "target": hf_save_dataset, 
            "private": hf_save_dataset_private,
            "audio_column": hf_save_dataset_column_audio, 
            "text_column": hf_save_dataset_column_text,
            "text_sbv_column": hf_save_dataset_column_text_sbv,
            "revision": hf_save_dataset_revision,
            #"subset": hf_save_dataset_subset,
            "split": hf_save_dataset_split
            }
    
    print(hf_load_dataset_options)

    whisper_pipeline = NeuraLumaWhisperPipeline(dtype=getattr(jnp, dtype), batch_size=batch_size, hf_checkpoint=hf_checkpoint, hf_load_dataset_options=hf_load_dataset_options)

    youtube_urls = youtube.split(";")

    whisper_pipeline.transcribe(source_path=source_path, output_path=output_path, hf_save_dataset_options=hf_save_dataset_options, youtube_urls=youtube_urls, add_timestamps=add_timestamps, translate=translate)
    