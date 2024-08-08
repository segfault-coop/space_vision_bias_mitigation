# space_vision_bias_mitigation

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