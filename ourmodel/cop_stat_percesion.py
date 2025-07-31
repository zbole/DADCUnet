import pandas as pd
import numpy as np

import glob

files=glob.glob(r'\\\/*/*/evaluation_results.csv')

for file in files:
    da=pd.read_csv(file)
    print('#'*100)
    print(file.split('\\')[4])
    print(da)



