# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 11:25:36 2019

@author: WNP387
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df_train = pd.read_csv(r"C:\Users\wnp387\Desktop\train.csv")

# Just looking at a single trial for now
subset = df_train.loc[(df_train['crew'] == 1) & (df_train['experiment'] == 'CA')]

subset.sort_values(by='time', inplace = True)


# Show the plot
x = subset['r'][3000:4024]
yaxes = range(3000,4024)
plt.plot(yaxes,x)


from scipy import signal

b, a = signal.butter(8,0.05)

y = signal.filtfilt(b, a, subset['r'], padlen=150)

plt.plot(yaxes,y[3000:4024])

