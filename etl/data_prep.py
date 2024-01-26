import numpy as np
import pandas as pd
import cv2
import yaml

# Get dataset from S3 if not exists in /data
# Read the files and preprocess - resize them

config = yaml.load('./config/dataset.yaml')

def prep_dataset():
    print(np.sin(10))
    return None

if __name__ == "__main__":
    prep_dataset()
