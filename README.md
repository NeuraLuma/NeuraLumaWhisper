# NeuraLuma Whisper
This repository aims to give an easy way of running Whisper with the following Features:
- Transcribe one or multiple Files (.mp3, .mp4)
- Transcribe one or multiple YouTube videos by URLs*
- As plain Text or optionaly as timestamped .sbv
- Usable as a CLI

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
- [ ] Support to load HF Datasets (Select two columns)
- [ ] Support to save HF Datasets (Audio -> Text Caption + Optional Timestamped caption)
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

## License
Please refer to the License file of this repository.

## Acknowledgements
This repository builds on the [Whisper JAX](https://github.com/sanchit-gandhi/whisper-jax/tree/main) Implementation from Sanchit Gandhi.