import pandas as pd
import numpy as np
import os
datadir = os.getcwd()
datadir = os.path.join(datadir, 'datasets')
for dirname, _, filenames in os.walk(datadir):
    for filename in filenames:
        print(os.path.join(dirname, filename))

