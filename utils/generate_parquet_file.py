import pandas as pd
import argparse
import regex as re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

parser = argparse.ArgumentParser(description="Generate a Parquet file from url")
parser.add_argument("--url", type=str, default="https://neo.gsfc.nasa.gov/archive/rgb/", help="Base URL of the site")
parser.add_argument("--output_file", type=str, default="data/MOD_LSTD_D.parquet", help="Output Parquet file")
parser.add_argument("--modality", type=str, default="MOD_LSTD_D", help="Modality of the images")
args = parser.parse_args()

url = args.url
output_file = args.output_file
modality = args.modality

date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')

def get_soup(url):
    response = requests.get(url)
    response.raise_for_status()
    return BeautifulSoup(response.text, 'html.parser')

def main():
    df = pd.DataFrame()
    soup = get_soup(url)
    links = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.startswith(modality):
            links.append(urljoin(url, href))

    for link in links:
        link_soup = get_soup(link)
        for m_link in link_soup.find_all('a'):
            m_href = m_link.get('href')
            if m_href and m_href.startswith(modality):
                download_link = urljoin(link, m_href)
                date = date_pattern.search(m_href).group(0)
                map_data = {
                    'modality': modality,
                    'date': date,
                    'download_link': download_link
                }
                df = pd.concat([df, pd.DataFrame([map_data])], ignore_index=True)

    df.to_parquet(output_file)
    print(f"Parquet file saved at {output_file}")

if __name__ == "__main__":
    main()
