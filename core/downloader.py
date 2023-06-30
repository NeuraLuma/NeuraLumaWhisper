from pytube import YouTube
import os

class YouTubeDownloader:
    def download(self, url: str, output_path: str = "temp/yt_downloads") -> str:
        """
        Downloads a file from the given URL and saves it to the specified output path.

        Parameters:
            url (str): The URL of the file to be downloaded.
            output_path (str): The path where the downloaded file will be saved. Defaults to "temp/yt_downloads".

        Returns:
            str: The path of the downloaded file.
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        yt = YouTube(url)
        saved_path = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').asc().first().download(output_path=output_path)

        return saved_path
    
