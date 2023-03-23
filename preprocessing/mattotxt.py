# File created to get a quick way to get the mat files to a text file for better visualization and in general just a good idea of what we're doing.

import scipy.io
import pandas as pd
from pathlib import Path
import numpy as np


path = Path("data/Healthy")

for child in path.iterdir():
    np.append(patharray)
