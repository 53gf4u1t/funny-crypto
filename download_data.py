import requests
import zipfile
import os

def download_and_extract(url, filename):
    os.makedirs("data", exist_ok=True)

    response = requests.get(url, stream=True)
    with open(filename + ".zip", "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

    with zipfile.ZipFile(filename + ".zip", 'r') as zip_ref:
        # Extract all contents into the target directory
        file = zip_ref.namelist()[0]
        zip_ref.extractall('.')

    os.remove(filename + ".zip")
    os.rename(file, f"data/{filename}.csv")

for year in range(2026, 2027):
    for month in range(1, 13 if year < 2026 else 7):
        download_and_extract(f"https://data.binance.vision/data/futures/um/monthly/klines/ETHUSDT/5m/ETHUSDT-5m-{year}-{month:02}.zip", f"{year}-{month:02}")