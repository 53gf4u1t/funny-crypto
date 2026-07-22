import pandas as pd
from pathlib import Path

files = sorted([f.name for f in Path("data").iterdir() if f.is_file()])
dataframe = pd.concat([
    pd.read_csv(f"data/{file}") for file in files
])

print(dataframe)