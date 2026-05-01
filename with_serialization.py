import kagglehub
import os
import pandas as pd
import model

# Loads in call center files into a df
path = kagglehub.dataset_download("oleksiymaliovanyy/call-center-transcripts-dataset")
file = os.listdir(path)
file = [f for f in file if f.endswith(".csv") and "description" not in f.lower()]
