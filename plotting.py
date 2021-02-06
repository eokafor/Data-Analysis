
import matplotlib.pyplot as plt
import numpy as np

# this file is for plotting functions only!

def plot_original_accel(df, filename):
    plt.figure()
    plt.xlabel('Time(s)')
    plt.ylabel('Unfiltered Total Acceleration')
    plt.title(filename + ' Total Acceleration')
    plt.plot(df['time'], df['atotal'], 'g-', linewidth=2)
    plt.savefig(filename+'_acceleration.png')
    plt.close()

def plot_filtered_accel(df, filename):
    plt.figure()
    plt.xlabel('Time(s)')
    plt.ylabel('Filtered Total Acceleration (m/s^2)')
    plt.title(filename + ' Filtered Total Acceleration')
    plt.plot(df['time'], df['atotal_filtered'], 'g-', linewidth=2)
    plt.savefig(filename+'_filtered_acceleration.png')
    plt.close()

def plot_velocity(df, filename):
    plt.figure()
    plt.xlabel('Time(s)')
    plt.ylabel('Velocity (m/s)')
    plt.title(filename + ' Velocity')
    plt.plot(df['time'], df['velocity'], 'g-', linewidth=2)
    plt.savefig(filename+'_velocity.png')
    plt.close()

def plot_position(df, filename):
    plt.figure()
    plt.xlabel('Time(s)')
    plt.ylabel('Position(m)')
    plt.title(filename + ' Position over time')
    plt.plot(df['time'], df['position'], 'g-', linewidth=2)
    plt.savefig(filename+'_position.png')
    plt.close()

def plot_fft(xf, yf, filename, pace):
    plt.figure()
    plt.xlabel('Frequency(Hz)')
    plt.xticks(np.arange(min(xf), max(xf)+1, 2.0), fontSize=8)
    plt.ylabel('Amplitude')
    plt.title(filename + ' Walking Frequencies')
    plt.plot(xf, np.abs(yf), 'g-', linewidth=2)
    index = np.where(xf==pace)
    plt.text(pace, np.abs(yf[index]), "Walking Frequency", fontsize=10)
    plt.savefig(filename+'_frequency.png')
    plt.close()
