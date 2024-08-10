import os
from typing import List
import cv2
import argparse

def downscale_img(img_src:str, dst_folder:str, downscale_sz:tuple[int,int])->bool:
    img = cv2.imread(img_src)
    downscaled = cv2.resize(img, downscale_sz)
    path_to_save = os.path.join(dst_folder, os.path.basename(img_src))
    return cv2.imwrite(path_to_save, downscaled)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downscale images in a folder")
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(__file__), "../data/"), help="Directory containing the images")
    parser.add_argument("--scale", type=int, nargs=2, default=(256, 256), help="Size to downscale the images to")
    args = parser.parse_args()

    data_dir:str = args.data_dir
    scale:tuple[int,int] = args.scale

    folders:List[str] = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    for folder in folders:
        print(f"Using folder: {folder}")
        files:List[str] = [f for f in os.listdir(os.path.join(data_dir, folder)) if os.path.isfile(os.path.join(data_dir, folder, f))]
        new_folder:str = os.path.join(data_dir, f"{folder}_downscaled")

        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        for file in files:
            print(f"Using file: {file}")
            downscale_img(os.path.join(data_dir, folder, file), new_folder, scale)
