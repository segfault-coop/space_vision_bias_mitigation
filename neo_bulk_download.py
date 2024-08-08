import argparse
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Create argument parser
parser = argparse.ArgumentParser(description="Bulk download images from NEO archive")
parser.add_argument("--base_url", type=str, default="https://neo.gsfc.nasa.gov/archive/rgb/", help="Base URL of the site")
parser.add_argument("--pattern", type=str, default="MOD_LSTD_D", help="Folder pattern to match")
parser.add_argument("--download_dir", type=str, default="data/neo_images", help="Directory to save downloaded files")
parser.add_argument("--cleanup", type=str, default="no", help="Delete existing files in the download directory")

# Parse arguments
args = parser.parse_args()

# Base URL of the site
base_url = args.base_url

# Folder pattern to match
pattern = args.pattern

# Directory to save downloaded files
download_dir = args.download_dir

cleanup = args.cleanup

if cleanup == "yes":
    print(f"Cleaning up the download directory: {download_dir}")
    for file in os.listdir(download_dir):
        file_path = os.path.join(download_dir, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
            

# Create the download directory if it doesn't exist
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

def get_soup(url):
    response = requests.get(url)
    response.raise_for_status()  
    # Ensure we notice bad responses
    return BeautifulSoup(response.text, 'html.parser')

def download_file(file_url, dest_folder, pbar=None):
    local_filename = os.path.join(dest_folder, os.path.basename(file_url))
    with requests.get(file_url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    pbar.update(1)

def process_folder(folder_url):
    folder_soup = get_soup(folder_url)
    download_tasks = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        file_links = folder_soup.find_all("a")
        with tqdm(total=len(file_links), desc="Downloading files", unit="file") as pbar:
            for file_link in file_links:
                file_href = file_link.get("href")
                insensitive_href = file_href.lower()
                if file_href and insensitive_href.endswith(('.jpg', '.png', '.tif', '.tiff')):
                    file_url = urljoin(folder_url, file_href)
                    download_tasks.append(executor.submit(download_file, file_url, download_dir, pbar))
            for task in as_completed(download_tasks):
                task.result()  # To catch exceptions if any

def main():
    soup = get_soup(base_url)
    folder_urls = []

    # Find all links that match the pattern
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.startswith(pattern):
            folder_url = urljoin(base_url, href)
            folder_urls.append(folder_url)
            print(f"Found folder: {folder_url}")

    # Process folders with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(process_folder, url): url for url in folder_urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                future.result()  # To catch exceptions if any
            except Exception as e:
                print(f"Error processing {url}: {e}")

if __name__ == "__main__":
    main()
