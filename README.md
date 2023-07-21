# NeuraLuma Whisper
This repository aims to give an easy way of running Whisper with the following Features:
- Transcribe one or multiple Files (.mp3, .mp4)
- Transcribe one or multiple YouTube videos by URLs*
- As plain Text or optionaly as timestamped .sbv
- Usable as a CLI
- Supports loading and saving Huggingface Datasets from and to hub

*Please refer to the YouTube ToS. When using this feature, you aknowledge that you have the rights to do so.

## ToDos
- [x] Transcribe Audio File(s)
- [x] Transcribe Video File(s)
- [x] Transcribe YouTube videos with URLs
- [x] Plain Text Transcription
- [x] Transcription with Timestamps (.sbv)
- [x] CLI
- [ ] WebUI
- [ ] Accept More Formats (Audio and Video Formats)
- [x] Fine-tuned control over used dtype in JAX
- [ ] Better Error handling (e.g. YouTube)
- [x] Support for batch-size
- [x] Support for alternative HuggingFace Checkpoints
- [ ] Add Post cleanup options (delete temp folder)
- [ ] Add more verbose logging / progress (CLI)
- [ ] Add name scheming option (output filename)
- [x] Support to load HF Datasets (Select one audio column, revision, split)
- [x] Support to save HF Datasets (Audio -> Text Caption + Optional Timestamped caption)
- [ ] Support to optionally push to hub and load/save from local
- [x] GPU instructions
- [x] CPU instructions
- [ ] TPU instructions
- [ ] Apple Silicon (Metal support)
- [ ] Better Documentation
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
TODO

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
TODO

### Windows (CPU)
TODO

## Usage
TODO

### Basic Options
e.g. Timestamp, Translate, Output path TODO

### Loading Audio / Video File(s) from a path
TODO

### Loading one or multiple YouTube videos
TODO

### Loading an Audio Huggingface Dataset
When accessing private datasets, make sure to login with your huggingface account via `huggingface-cli login` and paste your auth token.
If you do not have one, create one [here](https://huggingface.co/settings/tokens).

You can use the `-ld` or `TODO` argument to load a huggingface audio dataset.

Here is an example how to load a dataset from the user `myuser` with the name `my_dataset`:
```sh
python main.py -ld "myuser/my_dataset" -o /Users/lily/NeuraLuma/NeuraLumaWhisper/files_out/ -ts -d "bfloat16" -b "4" -hfc "openai/whisper-large"
```

If your datasets have splits you can add the name to load via the `-ldsp` or `TODO` argument e.g. `train`:
```sh
python main.py -ld "myuser/my_dataset" -ldsp "train" -o /Users/lily/NeuraLuma/NeuraLumaWhisper/files_out/ -ts -d "bfloat16" -b "4" -hfc "openai/whisper-large"
```

The default revision / branch to load is set to `main`, however you can change that with the `-ldr` or `TODO` argument e.g. `trunk`:
```sh
python main.py -ld "myuser/my_dataset" -ldsp "train" -o /Users/lily/NeuraLuma/NeuraLumaWhisper/files_out/ -ts -d "bfloat16" -b "4" -hfc "openai/whisper-large"
```

The default column name is set to `audio`, if it's different you can change it with the `-ldc` or `TODO` argument e.g. `some_audio`:
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

Repositories are private by default, you can change that by using `-sp` or `--hf_save_dataset_private`:
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

### Advanced Model Options
e.g. batchsize, dtype, other checkpoint TODO

## License
Please refer to the License file of this repository.

## Acknowledgements
This repository builds on the [Whisper JAX](https://github.com/sanchit-gandhi/whisper-jax/tree/main) Implementation from Sanchit Gandhi.