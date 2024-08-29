# space_vision_bias_mitigation
### Overview
This repository contains code for processing space climate datasets, particularly using Masked Autoencoders (MAEs) and Stable Diffusion Models to perform data inference, training, and baseline comparisons. The goal is to enhance climate data analysis, downscale space climate images, and generate meaningful insights through advanced machine learning models.

The repository is divided into notebooks and utility scripts that handle different tasks, such as model training, inference, data preparation, and dataset download automation.

## Neo Dataset
This script allows you to bulk download images from the NEO archive.

### Usage
You can run the script file with the following command:
```bash
python neo_bulk_download.py --base_url <base_url> --pattern <pattern> --download_dir <download_dir> --cleanup <cleanup>
```
#### Arguments
- `base_url`: The base URL of the NEO archive. For example, `https://neo.sci.gsfc.nasa.gov/archive/`.
- `pattern`: The pattern of the images you want to download. For example, `MOD_LSTD_D`.
- `download_dir`: The directory where the images will be downloaded.
- `cleanup`: Whether to remove the downloaded images that are not in the pattern. Default is `no`.

Example:
```bash
python neo_bulk_download.py --base_url https://neo.gsfc.nasa.gov/archive/rgb/ --pattern MOD_LSTD_D --download_dir data/neo_images
```

## Installation
#### 1. Clone the repository:
```bash
git clone https://github.com/your_username/space-climate-ml.git
cd space-climate-ml
```
#### 2. Install dependencies:
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
#### 3.Set up the dataset:
* Use neo_bulk_download.py to download the space climate datasets.
* Use downscale_data.py to downscale the images if necessary.

