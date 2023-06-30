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
    
    def download_multiple(self, urls: list, output_path: str = "temp/yt_downloads") -> (list[str], list[str]):
        """
        Downloads multiple files from a list of URLs and saves them to the specified output path.

        Parameters:
            urls (list): A list of URLs from which to download the files.
            output_path (str): The path to the directory where the downloaded files will be saved. Defaults to "temp/yt_downloads".

        Returns:
            tuple: A tuple containing two lists - saved_paths and failed_paths.
                - saved_paths (list): A list of the paths to the successfully downloaded files.
                - failed_paths (list): A list of the URLs that failed to download.
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        saved_paths = []
        failed_paths = []

        for url in urls:
            try:
                # Prevent Exception from causing downloads from stopping
                saved_paths.append(self.download(url, output_path=output_path))
            except:
                failed_paths.append(url)
                continue
        
        return saved_paths, failed_paths
