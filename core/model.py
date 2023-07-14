from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

class WhisperModel:
    """
    The Whisper Model to interact with Whisper.
    """
    def __init__(self, dtype=jnp.float16, batch_size=1, checkpoint="openai/whisper-large-v2"):
        self.pipeline = FlaxWhisperPipline(checkpoint, batch_size=batch_size, dtype=dtype)
    
    def transcribe(self, inputs, add_timestamps=True) -> str:
        """
        Transcribes the content of a file located at `file_path` using the pipeline.

        Parameters:
            inputs (np.ndarray or str or bytes or dict): The input data to be transcribed. It can be one of the following:
                - np.ndarray: A numpy array representing the audio waveform.
                - str: The path to an audio file.
                - bytes: The content of an audio file.
                - dict: A dictionary containing the following keys:
                    - "array" (np.ndarray): A numpy array representing the audio waveform.
                    - "sampling_rate" (int): The sampling rate associated with the audio waveform.
            add_timestamps (bool, optional): Whether to add timestamps to the transcribed text. Defaults to True.

        Returns:
            str: The transcribed text.
        """
        text = self.pipeline(inputs, return_timestamps=add_timestamps)

        return text
    
    def translate(self, inputs, add_timestamps=True) -> str:
        """
        Translates the content of a file located at `file_path` using the pipeline.
        
        Args:
            inputs (np.ndarray or str or bytes or dict): The input data to be transcribed. It can be one of the following:
                - np.ndarray: A numpy array representing the audio waveform.
                - str: The path to an audio file.
                - bytes: The content of an audio file.
                - dict: A dictionary containing the following keys:
                    - "array" (np.ndarray): A numpy array representing the audio waveform.
                    - "sampling_rate" (int): The sampling rate associated with the audio waveform.
            add_timestamps (bool, optional): Whether to include timestamps in the translated text. Defaults to True.
        
        Returns:
            str: The translated text.
        """
        text = self.pipeline(inputs, task="translate", return_timestamps=add_timestamps)

        return text
