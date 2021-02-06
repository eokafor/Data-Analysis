import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.fft import rfft, rfftfreq
from scipy.signal import savgol_filter, find_peaks
import scipy.integrate as integ
import seaborn
import glob
from pathlib import Path

from walkPaceProcessing import butter_filter
from plotting import *

def filterData(df):
    # # we're only using atotal for this project
    total_accel = df.drop(['ax', 'ay', 'az'], axis=1)
    # first and last few seconds are not walking related, lets drop 5 off each end
    total_accel.drop(total_accel.head(500).index, inplace=True)
    total_accel.drop(total_accel.tail(500).index, inplace=True)
    total_accel['atotal_filtered'] = butter_filter(total_accel['atotal'])
    return total_accel

def integrateData(df):
    df['velocity'] = integ.cumtrapz(df['atotal_filtered'], x=df['time'], initial=0)
    df['position'] = integ.cumtrapz(df['velocity'], x=df['time'], initial=0.0)
    return df

def calcTotalDistance(df):
    diff = df.diff().dropna() # get the changes between each row
    diff['position'] = np.abs(diff['position'])
    return diff['position'].sum()


def main():
    seaborn.set()
    input_dir = './1kmdata'
    input_files = glob.glob(input_dir+'/*.csv')

    for f in input_files:
        subject_accel = pd.read_csv(f, sep=",")
        filtered_accel = filterData(subject_accel)
        plotName = Path(f).stem
        ## UNCOMMENT FOR GENERATING PLOTS
        # plot_original_accel(filtered_accel, plotName)
        # plot_filtered_accel(filtered_accel, plotName)

        df_integrated = integrateData(filtered_accel)
        ## UNCOMMENT FOR GENERATING PLOTS
        # plot_velocity(df_integrated, plotName)
        # plot_position(df_integrated, plotName)

        # print the final distance travelled
        distance = calcTotalDistance(df_integrated)
        print(plotName+' total distance travelled: {:0.2f}m'.format(distance))
        

if __name__ == '__main__':
    main()