### File for Gradio App ###
from core.pipeline import NeuraLumaWhisperPipeline
import gradio as gr
import jax.numpy as jnp
import functools

## Functions ##
def sanitize_args_decorator(function_to_decorate):
    def sanitize(arg):
        if isinstance(arg, str) and not arg.strip():
            return None
        return arg

    @functools.wraps(function_to_decorate)
    def sanitized_args_function(*args, **kwargs):
        sanitized_args = tuple(sanitize(arg) for arg in args)
        sanitized_kwargs = {key: sanitize(value) for key, value in kwargs.items()}
        return function_to_decorate(*sanitized_args, **sanitized_kwargs)

    return sanitized_args_function

@sanitize_args_decorator
def handle_submit(
        input_file, 
        input_directory, 
        input_youtube, 
        load_dataset,
        load_dataset_subset,
        load_dataset_split,
        load_dataset_revision,
        load_dataset_column,
        output_directory,
        save_dataset,
        save_dataset_revision,
        save_dataset_split,
        save_dataset_private,
        save_dataset_column_audio,
        save_dataset_column_text,
        save_dataset_column_text_sbv,
        dtype_options, 
        checkpoint,
        batch_size,
        translate,
        add_timestamps,
        hf_token,
        progress = gr.Progress()
):          
    source_path = None
    if input_file:
        source_path = input_file.name
    
    output_path = output_directory
    dtype = dtype_options
    hf_checkpoint = checkpoint
    
    # Input directory will be weighted higher if both options are specified
    if input_directory:
        source_path = input_directory
    
    hf_load_dataset_options = None
    if load_dataset:
        hf_load_dataset_options = {
            "source": load_dataset, 
            "audio_column": load_dataset_column, 
            "revision": load_dataset_revision,
            "subset": load_dataset_subset,
            "split": load_dataset_split
            }
    
    hf_save_dataset_options = None
    if save_dataset:
        hf_save_dataset_options = {
            "target": save_dataset, 
            "private": save_dataset_private,
            "audio_column": save_dataset_column_audio, 
            "text_column": save_dataset_column_text,
            "text_sbv_column": save_dataset_column_text_sbv,
            "revision": save_dataset_revision,
            #"subset": hf_save_dataset_subset,
            "split": save_dataset_split
            }
    
    youtube = input_youtube
    
    if not youtube and not source_path and not load_dataset:
        raise gr.Error("Please specify either YouTube Urls and/or a Dataset and/or a File or a directory")
    
    if not output_path and not save_dataset:
        raise gr.Error("Please specify either an Output Directory or a Dataset")
    
    if youtube is None:
        youtube = ""
    
    def progress_cb(msg, progress_amount=0):
        progress(progress_amount, desc=f"{msg}")

    whisper_pipeline = NeuraLumaWhisperPipeline(
        dtype=getattr(jnp, dtype), 
        batch_size=batch_size, 
        hf_checkpoint=hf_checkpoint, 
        hf_load_dataset_options=hf_load_dataset_options,
        hf_token=hf_token,
        progress_cb=progress_cb
        )

    youtube_urls = youtube.split("\n")

    progress(0, desc="Starting...")

    whisper_pipeline.transcribe(source_path=source_path, 
                                output_path=output_path, 
                                hf_save_dataset_options=hf_save_dataset_options, 
                                youtube_urls=youtube_urls, 
                                add_timestamps=add_timestamps, 
                                translate=translate)
    
    return "Complete!"

## UI ##
with gr.Blocks() as iface:
    # Options for the checkboxes
    supported_dtypes =['float16', 'bfloat16', 'float32', 'float64']
    supported_input_filetypes = ['.mp3', '.mp4']

    gr.Markdown("# NeuraLumaWhisper")

    with gr.Tab(label="Inference"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""## Input Options
                                Multiple may be used at once. 

                                Note: Only a File or Directory may be used (Directory is preferenced if both are used).

                                Leave options blank to not use them.""")
                    with gr.Tab(label="File"):
                        input_file = gr.File(label=f"Input File ({', '.join(supported_input_filetypes)})", file_types=supported_input_filetypes)

                    with gr.Tab(label="Directory"):
                        input_directory = gr.Textbox(label="Directory to load from", interactive=True)

                    with gr.Tab(label="YouTube"):
                        input_youtube = gr.Textbox(
                            label="YouTube URLs", 
                            info="Add URLs of the YouTube videos to be transcribed, put each URL in a new line", 
                            lines=10, 
                            multiline=True, 
                            interactive=True)

                    with gr.Tab(label="Huggingface Dataset"):
                        load_dataset = gr.Textbox(label="Dataset path", interactive=True)
                        load_dataset_subset = gr.Textbox(label="Dataset subset", interactive=True)
                        load_dataset_split = gr.Textbox(label="Dataset split", interactive=True)
                        load_dataset_revision = gr.Textbox(label="Dataset revision / branch", interactive=True, value="main")
                        load_dataset_column = gr.Textbox(label="Dataset column name for audio", interactive=True, value="audio")
                
                with gr.Column():
                    gr.Markdown("""## Output Options
                                Multiple may be used at once. 
                                
                                Leave options blank to not use them.
                                
                                <br/>""")

                    with gr.Tab(label="Directory"):
                        output_directory = gr.Textbox(label="Output Directory to save to", interactive=True)

                    with gr.Tab(label="Huggingface Dataset"):
                        save_dataset = gr.Textbox(label="Dataset path", interactive=True)
                        save_dataset_revision = gr.Textbox(label="Dataset revision / branch", interactive=True, value="main")
                        #save_dataset_subset = gr.Textbox(label="Dataset subset", interactive=True)
                        save_dataset_split = gr.Textbox(label="Dataset split", interactive=True)
                        save_dataset_private = gr.Checkbox(label="Private", interactive=True, value=True)
                        save_dataset_column_audio = gr.Textbox(label="Dataset column name for audio", interactive=True, value="audio")
                        save_dataset_column_text = gr.Textbox(label="Dataset column name for normal transcription", interactive=True, value="text")
                        save_dataset_column_text_sbv = gr.Textbox(label="Dataset column name for timestamped transcription", 
                                                                  interactive=True, value="sbv")
                
            submit_button = gr.Button("Transcribe", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
            
    with gr.Tab(label="Model Options"):
        with gr.Row():
            dtype_options = gr.Dropdown(choices=supported_dtypes, 
                                        label="Model dtype", 
                                        info="Recommended: bfloat16 (e.g. A100 / TPU) or float16 (most users)",
                                        value='float16', 
                                        interactive=True)
            checkpoint = gr.Textbox(label="Model Checkpoint", 
                                    value='openai/whisper-large-v2', 
                                    info="Path to the model checkpoint, must be compatible with the JAX implementation",
                                    interactive=True)
        
        with gr.Row():
            batch_size = gr.Number(label="Batch size", value=1, minimum=1, interactive=True)

        with gr.Row():
            with gr.Column():
                translate = gr.Checkbox(
                    label="Translate", 
                    info="Translates the audio to english regardless of source language", 
                    value=False)
                add_timestamps = gr.Checkbox(label="Add Additional Timestamps", 
                                             info="""Uses sbv formatting for additional timestamps. 
                                             A seperate file / column for Huggingface Datasets will be created""", 
                                             value=False)
        
    with gr.Tab(label="Huggingface Authentication"):
        hf_token = gr.Textbox(label="Huggingface Token", value="", 
                            info="""Token for fetching private models and datasets and pushing to hub.
                            Only necessary if not authenticated already via huggingface-cli login""", 
                            type="password",
                            interactive=True)
    
    submit_button.click(fn=handle_submit, 
        inputs=[
            input_file, 
            input_directory, 
            input_youtube, 
            load_dataset,
            load_dataset_subset,
            load_dataset_split,
            load_dataset_revision,
            load_dataset_column,
            output_directory,
            save_dataset,
            save_dataset_revision,
            save_dataset_split,
            save_dataset_private,
            save_dataset_column_audio,
            save_dataset_column_text,
            save_dataset_column_text_sbv,
            dtype_options, 
            checkpoint,
            batch_size,
            translate,
            add_timestamps,
            hf_token
            ], 
        outputs=[status])

# Launch the interface
iface.queue().launch()
