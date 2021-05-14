import sys
sys.path.append('/Users/daiyiluo/Documents/ELEC599')
import numpy as np
import matplotlib.pyplot as plt
import CalHMM
import warnings
warnings.filterwarnings('ignore')
import HMMCrossValid as hcv

CA = 'KQ086_210110' #KQ095_210205' #
folder = f'/home/della/Downloads/Fish_data/{CA}/tb_500'
timebin = 500
name = 'Trace'
n_chunks = 6
k = 2
win = 2
n_iter = 20
self = hcv.datainfo(folder, timebin, name, n_chunks, k, win,normalize=False)

i_iter=0
while i_iter<n_iter:
    try:
        self.GaussianSetCrossValid(text=False)
        i_iter+=1
    except:
        print('failed')
        pass

CalHMM.save_object(self,f'/home/della/Downloads/Fish_data/cv/{self.name}_{CA}')


print(f'{self.name}, {self.folder}')
print(f'lap_range: {self.lap_range}\nn_laps_perchunk:{self.n_laps_perchunk}')
