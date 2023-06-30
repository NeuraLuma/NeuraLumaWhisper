from core.downloader import YouTubeDownloader
from moviepy.editor import AudioFileClip
import os

class AudioConverter:
    def convert_from_youtube(self, url: str, output_path: str = 'temp/audio', output_file_name: str|None = None) -> str:
        """
        Converts a YouTube video to an audio file.

        Parameters:
            url (str): The URL of the YouTube video.
            output_path (str): The path where the output audio file will be saved. Defaults to 'temp/audio'.
            output_file_name (str|None): The name of the output audio file. If None, a default name will be generated based on the video filename. Defaults to None.

        Returns:
            str: The path to the saved output audio file.
        """
        downloader = YouTubeDownloader()
        saved_video_path = downloader.download(url)

        # separate the directory and filename
        saved_video_dir = os.path.dirname(saved_video_path)
        saved_video_filename = os.path.basename(saved_video_path)

        if output_file_name is None:
            output_file_name = os.path.splitext(saved_video_filename)[0] + '.mp3'

        output_saved_path = self.convert_from_video(saved_video_dir, saved_video_filename, output_path, output_file_name)

        return output_saved_path
    def convert_from_video(self, input_path: str, input_file_name: str, output_path: str = 'temp/audios', output_file_name: str = 'audio.mp3') -> str:
        """
        Converts a video file to an audio file.

        Args:
            input_path (str): The path to the directory containing the input video file.
            input_file_name (str): The name of the input video file.
            output_path (str, optional): The path to the directory where the output audio file will be saved. Defaults to 'temp/audios'.
            output_file_name (str, optional): The name of the output audio file. Defaults to 'audio.mp3'.

        Returns:
            str: The path to the converted audio file.
        """
        input_source_path = os.path.join(input_path, input_file_name)
        output_destination_path = os.path.join(output_path, output_file_name)

        if not os.path.exists(input_source_path):
            raise Exception(f"Could not locate file at: {input_source_path}")
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with AudioFileClip(input_source_path) as clip:
            clip.write_audiofile(output_destination_path, codec='mp3')
        
        return output_destination_path