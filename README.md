# NeuraLuma Whisper
This repository aims to give an easy way of running Whisper with the following Features:
- Transcribe one or multiple Files (.mp3, .mp4)
- Transcribe one or multiple YouTube videos by URLs*
- As plain Text or optionaly as timestamped .sbv
- Usable as a CLI & WebUI
- Supports loading and saving Huggingface Datasets from and to hub

*Please refer to the YouTube ToS. When using this feature, you aknowledge that you have the rights to do so.

## Disclaimer
This repository is not perfect and not finished. There are still a lot of potential improvements that could be made. The repository could be seen in an "Experimental" Stage that has been created to prototype a potential tool that we might build upon in the future. We wanted to share our progress, so you might be able to use it, build upon it, or experiment with it as you wish.
New commits to this repository might introduce breaking changes and break existing code, so use with caution.

## ToDos
- [x] Transcribe Audio File(s)
- [x] Transcribe Video File(s)
- [x] Transcribe YouTube videos with URLs
- [x] Plain Text Transcription
- [x] Transcription with Timestamps (.sbv)
- [x] CLI
- [x] WebUI
- [ ] Accept More Formats (Audio and Video Formats)
- [x] Fine-tuned control over used dtype in JAX
- [ ] Better Error handling (e.g. YouTube)
- [x] Support for batch-size
- [x] Support for alternative HuggingFace Checkpoints
- [x] Add Post cleanup options (delete temp folder)
- [x] Add more verbose logging / progress (CLI)
- [ ] Improve Logging Progress
- [ ] Add name scheming option (output filename)
- [x] Support to load HF Datasets (Select one audio column, revision, split)
- [x] Support to save HF Datasets (Audio -> Text Caption + Optional Timestamped caption)
- [ ] Add option to save subsets and not just load them
- [ ] Support to optionally push to hub and load/save from local
- [ ] Simplify useage with code directly, so it can be used in pipelines
- [ ] Improve Error Handling and introduce checkpoints so progress may not be lost in a late error
- [ ] Improve Temp Folder Implementation, as it depends on execution context and might create temp folders when executing outside directory
- [ ] Make more flexible so other Models / Implementations can be used (e.g. the original HF implementation)
- [ ] UI: Add direct text output option
- [ ] Add option to provide multiple file-paths instead of just one file or one directory
- [ ] Implement yielding of chunks so output can be streamed
- [x] GPU instructions
- [x] CPU instructions
- [ ] TPU instructions
- [ ] Apple Silicon (Metal support)
- [x] Documentation CLI
- [ ] Documentation WebUI
- [ ] Installation Instructions for Windows

## Prerequisites
Please make sure to have Conda / Miniconda installed. We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
Alternatively, you can install the dependencies using another environment or directly (this documentation only supports Conda).

### Installation on Linux (64bit):
```sh
wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ./Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh 
source ~/.bashrc
conda config --set auto_activate_base false
conda deactivate
```

### Installation on Linux (ARM):
```sh
wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
chmod +x ./Miniconda3-latest-Linux-aarch64.sh
./Miniconda3-latest-Linux-aarch64.sh 
source ~/.bashrc
conda config --set auto_activate_base false
conda deactivate
```

### Installation on MacOS (Apple Silicon):
Make sure commandline Tools are installed: `xcode-select --install`
```sh
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
chmod +x ./Miniconda3-latest-MacOSX-arm64.sh
./Miniconda3-latest-MacOSX-arm64.sh 
source ~/.bashrc
conda config --set auto_activate_base false
conda deactivate
```

### Installation on MacOS (Intel):
Make sure commandline Tools are installed: `xcode-select --install`
```sh
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
chmod +x ./Miniconda3-latest-MacOSX-x86_64.sh
./Miniconda3-latest-MacOSX-x86_64.sh 
source ~/.bashrc
conda config --set auto_activate_base false
conda deactivate
```

### Installation on Windows:
Refer to this [Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html).

## Installation

### Linux (GPU)
```sh
conda env create -f environmentGPU.yaml
conda activate NeuraLumaWhisperGPU
```
### Linux (TPU)
TODO

### MacOS (Apple Silicon) - Currently not working
Note: As JAX has to be installed and built from Source to be supported with Apple Silicon it takes a while. The Implementation is also Experimental and may cause unexpected behaviour.

**Currently jax-metal does not support `mhlo.convolution` which is used in the jax implementation**

```sh
chmod +x ./installOSX_Silicon.sh
./installOSX_Silicon.sh
conda activate NeuraLumaWhisperSilicon
```

### MacOS / Linux (CPU)
```sh
conda env create -f environmentCPU.yaml
conda activate NeuraLumaWhisperCPU
```

### Windows (GPU)
Make sure to open a Powershell Terminal.
```sh
conda env create -f environmentGPU.yaml
conda activate NeuraLumaWhisperGPU
```

### Windows (CPU)
Make sure to open a Powershell Terminal.
```sh
conda env create -f environmentCPU.yaml
conda activate NeuraLumaWhisperCPU
```

### Basic Options
`-tl`, `--translate`: Activates translation mode. All audio will automatically be transcribed to english.
`-ts`, `--timestamp`: Activates timestamps. Adds a separate file with sbv extension / add a column in the HF Dataset.
`-o`, `--output`: Path to the directory for the transcribed files.

### Loading Audio / Video File(s) from a path
You can load audio or video files from a path using the `-s` or `--source` argument.

Loading a single audio file:
```sh
python main.py -s "/path/to/audio/file.mp3" -o /path/to/output/
```
or a video file:
```sh
python main.py -s "/path/to/video/file.mp4" -o /path/to/output/
```

You may also provide a path to a directory with your audio and video files. All nested folders will be automatically searched for compatible audio and video files:
```sh
python main.py -s "/path/to/files" -o /path/to/output/
```

### Loading one or multiple YouTube videos
For loading YouTube videos, use the `-y` or `--youtube` argument. For multiple videos, separate the URLs with semicolons.

```sh
python main.py -y "youtube.com/watch?v=abcd;youtube.com/watch?v=efgh" -o /path/to/output/
```

### Loading an Audio Huggingface Dataset
When accessing private datasets, make sure to login with your huggingface account via `huggingface-cli login` and paste your auth token.
If you do not have one, create one [here](https://huggingface.co/settings/tokens).

You can use the `-ld` or `--hf_load_dataset` argument to load a huggingface audio dataset.

Here is an example how to load a dataset from the user `myuser` with the name `my_dataset`:
```sh
python main.py -ld "myuser/my_dataset" -o /Users/lily/NeuraLuma/NeuraLumaWhisper/files_out/ -ts -d "bfloat16" -b "4" -hfc "openai/whisper-large"
```

If your datasets have splits you can add the name to load via the `-ldsp` or `--hf_load_dataset_split` argument e.g. `train`:
```sh
python main.py -ld "myuser/my_dataset" -ldsp "train" -o /Users/lily/NeuraLuma/NeuraLumaWhisper/files_out/ -ts -d "bfloat16" -b "4" -hfc "openai/whisper-large"
```

The default revision / branch to load is set to `main`, however you can change that with the `-ldr` or `--hf_load_dataset_revision` argument e.g. `trunk`:
```sh
python main.py -ld "myuser/my_dataset" -ldsp "train" -o /Users/lily/NeuraLuma/NeuraLumaWhisper/files_out/ -ts -d "bfloat16" -b "4" -hfc "openai/whisper-large"
```

The default column name is set to `audio`, if it's different you can change it with the `-ldc` or `--hf_load_dataset_column` argument e.g. `some_audio`:
```sh
python main.py -ld "myuser/my_dataset" -ldsp "train" -ldc "some_audio" -o /Users/lily/NeuraLuma/NeuraLumaWhisper/files_out/ -ts -d "bfloat16" -b "4" -hfc "openai/whisper-large"
```

Selecting subsets is also supported. Let's say you have subsets for different locales, e.g. `en-GB`. You can select it using the `-ldst` or `--hf_load_dataset_subset` argument:
```sh
python main.py -ld "myuser/my_dataset" -ldsp "train" -ldc "some_audio" -ldst "en-GB" -o /Users/lily/NeuraLuma/NeuraLumaWhisper/files_out/ -ts -d "bfloat16" -b "4" -hfc "openai/whisper-large"
```

### Saving transcriptions as an Audio / Text Huggingface Dataset
When trying to push to HuggingfaceHub, make sure to login with your huggingface account via `huggingface-cli login` and paste your auth token.
If you do not have one, create one [here](https://huggingface.co/settings/tokens).

To create a audio to text dataset you simply have to supply a repository path of the huggingface repository that follows the `user/name` namespace. Here is an Example to push to `example_user`'s `example` Dataset using the `-sd` or `--hf_save_dataset` argument:

```sh
python main.py -s "/Users/lily/NeuraLuma/NeuraLumaWhisper/files_in/" -o /Users/lily/NeuraLuma/NeuraLumaWhisper/files_out/ -ts -d "bfloat16" -b "4" -sd "example_user/example"
```

Repositories are private by default, you can change that by using `-sdp` or `--hf_save_dataset_private`:
```sh
python main.py -s "/Users/lily/NeuraLuma/NeuraLumaWhisper/files_in/" -o /Users/lily/NeuraLuma/NeuraLumaWhisper/files_out/ -ts -d "bfloat16" -b "4" -sd "example_user/example" -sp False
```

You can also adjust the column names for the input audio (default `audio`), the output text transcription (default `text`) and the output text transcription with timestamps (default `sbv`) by using these arguments:

- `-sdca` or `--hf_save_dataset_column_audio` for the column name of the input audio
- `-sdct` or `--hf_save_dataset_column_text` for the column name of the transcribed text
- `-sdcsbv` or `--hf_save_dataset_column_text_sbv` for the column name of the transcribed text with timestamps in sbv formatting

e.g. setting the audio column name to `audio_in`, the text column name to `transcription_plain` and the timestamped text to `transcription_timestamped`:
```sh
python main.py -s "/Users/lily/NeuraLuma/NeuraLumaWhisper/files_in/" -o /Users/lily/NeuraLuma/NeuraLumaWhisper/files_out/ -ts -d "bfloat16" -b "4" -sd "example_user/example" -sdca "audio_in" -sdct "transcription_plain" -sdcsbv "transcription_timestamped"
```

Also you may change the revision / branch name on huggingface by specifying `-sdr` or `--hf_save_dataset_revision` e.g. to `trunk`:
```sh
python main.py -s "/Users/lily/NeuraLuma/NeuraLumaWhisper/files_in/" -o /Users/lily/NeuraLuma/NeuraLumaWhisper/files_out/ -ts -d "bfloat16" -b "4" -sd "example_user/example" -sdr "trunk"
```

You can also specify a split by using `-sdsp` or `--hf_save_dataset_split`:
```sh
python main.py -s "/Users/lily/NeuraLuma/NeuraLumaWhisper/files_in/" -o /Users/lily/NeuraLuma/NeuraLumaWhisper/files_out/ -ts -d "bfloat16" -b "4" -sd "example_user/example" -sdr "trunk" -sdsp "train"
```

### Advanced Model Options
**Batch Size**: Sets the batch size for inference. The batch size can significantly impact the speed and memory usage of model inference. Larger batch sizes allow the model to process more data at once, but require more memory. Conversely, smaller batch sizes use less memory but may take longer to process the same amount of data. Also note that, the Word Error Rate may increase slightly depending on the Batch Size. You can set the batch size with `-b` or `--batch_size`. Default is `1`. Example:
```sh
python main.py -b "4" ...
```

**Data Type (dtype)**: Specifies the data type to use. Different data types use different amounts of memory and have different levels of numerical precision. Options include `float16`, `bfloat16`, `float32`, and `float64`. You can set the data type with `-d` or `--dtype`. Default is `float16`. We recommend using either `float16` (for most users) or using `bfloat16` (for example TPU / A100 users). Example:
```sh
python main.py -d "bfloat16" ...
```

**Hugging Face Checkpoint**: Specifies the pre-trained model to use for inference. The model is specified by the name of its checkpoint on the Hugging Face model hub. You can set the checkpoint with `-hfc` or `--hf_checkpoint`. Default is `openai/whisper-large-v2`. Example:
```sh
python main.py -hfc "openai/whisper-large" ...
```

## License
Please refer to the License file of this repository.

## Acknowledgements
This repository builds on the [Whisper JAX](https://github.com/sanchit-gandhi/whisper-jax/tree/main) Implementation from Sanchit Gandhi.