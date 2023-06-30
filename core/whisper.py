from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

class Whisper:
    def __init__(self, dtype=jnp.float16, batch_size=1, checkpoint="openai/whisper-large-v2"):
        self.pipeline = FlaxWhisperPipline(checkpoint, batch_size=batch_size)
        self.dtype = dtype # Dtype should be set to jnp.float16 for most GPUs, for A100 or TPU set to jnp.bfloat16
    
    def transcribe(self, file_path, add_timestamps=True) -> str:
        """
        Transcribes the content of a file located at `file_path` using the pipeline.

        Parameters:
            file_path (str): The path of the file to be transcribed.
            add_timestamps (bool, optional): Whether to add timestamps to the transcribed text. Defaults to True.

        Returns:
            str: The transcribed text.
        """
        text = self.pipeline(file_path, dtype=self.dtype, return_timestamps=add_timestamps)

        return text
    
    def translate(self, file_path, add_timestamps=True) -> str:
        """
        Translates the content of a file located at `file_path` using the pipeline.
        
        Args:
            file_path (str): The path to the file to be translated.
            add_timestamps (bool, optional): Whether to include timestamps in the translated text. Defaults to True.
        
        Returns:
            str: The translated text.
        """
        text = self.pipeline(file_path, dtype=self.dtype, task="translate", return_timestamps=add_timestamps)

        return text
    