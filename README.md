# NeuraLuma Whisper
This repository aims to give an easy way of running Whisper with the following Features:
- TODO: Transcribe multiple Audio Files
  - TODO: Multiple formats
- TODO: Transcribe multiple YouTube URLs
- TODO: Transcribe Videos (convert to Audio First)
- TODO: Create text files
- TODO: Create caption files
- TODO: Annotate Videos with captions
- TODO: Create a Huggingface Dataset with upload functionality
- TODO: Offers a CLI
- TODO: Offers a UI

## Prerequisites
Please make sure to have Conda / Miniconda installed. We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### Installation on Linux (64bit):
```sh
wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ./Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b
source ~/.bashrc
conda config --set auto_activate_base false
conda deactivate
```

### Installation on Linux (ARM):
```sh
wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
chmod +x ./Miniconda3-latest-Linux-aarch64.sh
./Miniconda3-latest-Linux-aarch64.sh -b
source ~/.bashrc
conda config --set auto_activate_base false
conda deactivate
```

### Installation on MacOS (Apple Silicon):
Make sure commandline Tools are installed: `xcode-select --install`
```sh
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
chmod +x ./Miniconda3-latest-MacOSX-arm64.sh
./Miniconda3-latest-MacOSX-arm64.sh -b
source ~/.bashrc
conda config --set auto_activate_base false
conda deactivate
```

### Installation on MacOS (Intel):
Make sure commandline Tools are installed: `xcode-select --install`
```sh
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
chmod +x ./Miniconda3-latest-MacOSX-x86_64.sh
./Miniconda3-latest-MacOSX-x86_64.sh -b
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

## Useage
TODO

## License
Please refer to the License file of this repository.

## Acknowledgements
This repository builds on the [Whisper JAX](https://github.com/sanchit-gandhi/whisper-jax/tree/main) Implementation from Sanchit Gandhi.