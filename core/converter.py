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
        saved_video_path = downloader.download(url=url)

        # separate the directory and filename
        saved_video_dir = os.path.dirname(saved_video_path)
        saved_video_filename = os.path.basename(saved_video_path)

        if output_file_name is None:
            output_file_name = os.path.splitext(saved_video_filename)[0] + '.mp3'

        output_saved_path = self.convert_from_video(
            input_path=saved_video_dir,
            input_file_name=saved_video_filename, 
            output_path=output_path, 
            output_file_name=output_file_name
            )

        return output_saved_path
    def convert_multiple_from_youtube(self, urls: list, output_path: str = 'temp/audio', output_file_names: list|None = None) -> (list[str], list[str]):
        """
        Converts multiple YouTube videos to audio files.

        Args:
            urls (list): A list of YouTube video URLs.
            output_path (str, optional): The path to save the output audio files. Defaults to 'temp/audio'.
            output_file_names (list|None, optional): A list of output file names. If None, the file names are generated automatically. Defaults to None.

        Returns:
            tuple: A tuple containing two lists:
                - output_saved_paths (list[str]): A list of paths where the output audio files are saved.
                - failed_video_urls (list[str]): A list of YouTube video URLs that failed to convert.
        """
        downloader = YouTubeDownloader()
        saved_video_paths, failed_video_urls = downloader.download_multiple(urls=urls)

        # separate the directory and filename
        output_file_names = []

        for saved_video_path in saved_video_paths:
            saved_video_filename = os.path.basename(saved_video_path)
            output_file_names.append(os.path.splitext(saved_video_filename)[0] + '.mp3')
        
        output_saved_paths = self.convert_multiple_from_videos(
            input_source_paths=saved_video_paths,
            output_path=output_path,
            output_file_names=output_file_names
            )

        return output_saved_paths, failed_video_urls
        
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
    
    def convert_multiple_from_videos(self, input_source_paths: list[str], output_path: str = 'temp/audios', output_file_names: list|None = None) -> list[str]:
        """
        Convert multiple videos to audio files.

        Parameters:
            input_paths (list[str]): A list of input video file paths.
            output_path (str, optional): The output directory path for the audio files. Defaults to 'temp/audios'.
            output_file_names (list[str], optional): A list of output file names for the audio files. Defaults to None.

        Returns:
            list[str]: A list of output file paths for the converted audio files.
        """
        output_destination_paths = []
        
        if output_file_names is None:
            for idx, input_source_path in enumerate(input_source_paths):
                output_file_name = 'audio_' + str(idx) + '.mp3'
                output_destination_paths.append(os.path.join(output_path, output_file_name))
        else:
            for output_file_name in output_file_names:
                output_destination_paths.append(os.path.join(output_path, output_file_name))
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        for input_source_path in input_source_paths:
            if not os.path.exists(input_source_path):
                raise Exception(f"Could not locate file at: {input_source_path}")
        
        for input_source_path, output_destination_path in zip(input_source_paths, output_destination_paths):
            with AudioFileClip(input_source_path) as clip:
                clip.write_audiofile(output_destination_path, codec='mp3')
        
        return output_destination_paths