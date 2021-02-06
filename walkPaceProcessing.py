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
from plotting import plot_fft # file contains only functions to generate plots



SAMPLE_RATE = 100 #given by the app

def butter_filter(data):
    # bandpass filter that removes noise above 3hz and below 0.5hz to get rid of the 0.1hz signal we saw that we think represents changes in the overall acceleration (slowing/speeding)
    b, a = signal.butter(3, [0.5, 3], btype='bandpass', analog=False, fs=100)
    y = signal.filtfilt(b, a, data)
    return y

def calc_fft(data):
    yf = rfft(data)
    xf = rfftfreq(data.size, 1. / SAMPLE_RATE)
    return yf, xf

def get_walking_pace(df, name):
    # # we're only using atotal for this project
    total_accel = df.drop(['ax', 'ay', 'az'], axis=1)
    # first and last few seconds are not walking related, lets drop 5 off each end
    total_accel.drop(total_accel.head(500).index, inplace=True)
    total_accel.drop(total_accel.tail(500).index, inplace=True)
    # filter singal through bandpass
    filtered_accel = butter_filter(total_accel['atotal'])
    # perform rfft
    amp, freq = calc_fft(filtered_accel)

    ## in order to get the walking pace, lets get the top 2 dominant frequencies
    ## 1 will be the signal for step to step of a single foot, the other will be the pace of each step (what we want)
    ## we want the largest of these frequencies
    # get indices of all significant peaks of our data
    peaks, _ = find_peaks(np.abs(amp), height=1250) # this minimum height works well for our plots
    # translate peak indices to frequencies
    peak_freqs = []
    for i in peaks:
        peak_freqs.append(freq[i])
    peak_freqs = [x for x in peak_freqs if x>1.1 and x<2.3] # by analyzing the plots we know that the walking paces all fall within this range
    pace = max(peak_freqs)

    ## UNCOMMENT TO GENERATE PLOTS
    # plot_fft(freq, amp, name, pace)

    return pace

def main():
    seaborn.set()

    input_dir = './1minData'
    freqs_df = pd.read_csv('subject_gender.csv', sep=',')
    all_paces = np.zeros(freqs_df.shape[0]) # to be filled by processing the data
    # for all subjects, read in the data, filter, fft and calculate the walking pace
    for i in range(1,32):
        subject_accel = pd.read_csv(input_dir+'/'+'subject'+str(i)+'.csv', sep=",")
        name = 'subject'+str(i)
        pace = get_walking_pace(subject_accel, name)
        all_paces[i-1] = pace
    freqs_df['walking_pace'] = all_paces
    freqs_df['steps_per_min'] = freqs_df['walking_pace'] * 60
    freqs_df.to_csv('subject_walk_pace_results.csv', sep=',', index=False)
    
if __name__ == '__main__':
    main()