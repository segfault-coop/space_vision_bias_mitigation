import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import datasets
import pandas as pd
from io import BytesIO

def process_PIL_to_CV2(img):
    img = np.array(img, dtype=np.uint8)
    img = img[:, :, ::-1].copy()
    return img

def load_images_from_folder(dataset_name="data/neo_images_downscaled"):
    data_imgs = datasets.load_dataset(dataset_name)
    images = data_imgs['train']['image']
    images = [process_PIL_to_CV2(img) for img in images]
    print(f'Total number of imgs {len(images)}')
    return images


def create_mask(image, lower, upper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def calculate_overlap(mask, database_image):
    masked_image = cv2.bitwise_and(database_image, database_image, mask=mask)
    overlap_pixels = cv2.countNonZero(cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY))
    mask_pixels = cv2.countNonZero(mask)
    return overlap_pixels / mask_pixels

def save_usable_images(images, target_mask, threshold=0.7, save_dir="data/neo_usable"):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

basefile = "Baseline.png"
save_dir="data/neo_usable"
invariant_name_basefile = os.path.join(os.path.dirname(__file__), basefile) 
target_image = cv2.imread(invariant_name_basefile)
target_downscale = cv2.resize(target_image, (256,256))
lower = np.array([0, 0, 10])
upper = np.array([179, 255, 255])
target_mask = create_mask(target_image, lower, upper)
target_mask = cv2.resize(target_mask, (256,256))
database_images = load_images_from_folder()
threshold = 0.7
save_usable_images(database_images, target_mask, threshold=threshold, save_dir="data/neo_usable")

matched_images = []
for idx, db_image in enumerate(database_images):
    db_mask = create_mask(db_image, lower, upper)
    overlap_percentage = calculate_overlap(target_mask, db_image)
    if overlap_percentage >= threshold:
        matched_images.append((idx, db_image, overlap_percentage))
        save_path = os.path.join(save_dir, f"usable_img_{idx}.png")
        cv2.imwrite(save_path, db_image)  # Save the image
  

imgs_useable_idx = pd.DataFrame(columns=['hf_idx'])


if matched_images:
    print("Images that meet the threshold:")
    for idx, img, overlap in matched_images:
        print(f'Overlap --- {overlap}')
        imgs_useable_idx.loc[len(imgs_useable_idx)] = [idx]
else:
    print("No images meet the threshold.")
    
print(f'Usabel imgs {len(imgs_useable_idx)}')
imgs_useable_idx.to_csv("local_ds.csv")
