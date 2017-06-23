import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from parse_raw_data import obtain_raw_data
from create_model import LeadModel

# codes to set: 15, 260, 283, 235

# set global params
param_dict = {'code': 283, 'plot_raw': False, 'get_data': False, 'Thr_hist': [3, 7, 10], 'nu': 0.9, 'method': 'Hist',
              'numF_ir': 5}

# obtain raw data from files if it is required

if param_dict.get('get_data'):
    valid_files_to_process = obtain_raw_data(param_dict)
    print valid_files_to_process.head(5)

# process data
myModel = LeadModel(param_dict)
test_pd = myModel.do_model()

print test_pd



