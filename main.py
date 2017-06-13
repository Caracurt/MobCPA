import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from parse_raw_data import obtain_raw_data

# testing stuff
if False:
    N = 1000
    x = np.random.normal(0.0, 1.0, size=(1, N))
    y = np.random.normal(0.0, 2.0, size=(1, N))

    plt.scatter(x, y, c='r')
    plt.show()

# set global params

# obtain raw data from files
pd_test1 = obtain_raw_data()
print pd_test1.head(100)

# process data


