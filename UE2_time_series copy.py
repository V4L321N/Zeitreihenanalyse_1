# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 17:19:26 2023

@author: flori
"""

import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


head = list(pd.DataFrame(data=pd.read_csv("Rec040.csv")))
dataset040 = pd.DataFrame(data=pd.read_csv("Rec040.csv"))
locTemp040 = dataset040.loc[:, 'Local Temp [degC]']

def temperature_FFT(item):
    return np.fft.fft(item, norm="ortho")

fft_dataset = temperature_FFT(locTemp040)
fft_dataset = fft_dataset[5:]

locdataset = locTemp040

"""#load data
head = list(pd.DataFrame(data=pd.read_csv("ClimateData.csv")))
dataset = pd.DataFrame(data=pd.read_csv("ClimateData.csv"))
locdataset = dataset.loc[:, 'Globe']
# print(locdataset)
# locdataset = locdataset[1:]

len_data = len(locdataset)
print('number of data points = '+ str(len_data))"""


# Furier Trafo von Datenset
fft_dataset = np.fft.fft(locdataset) #/ np.sqrt(len_data)
fft_dataset = fft_dataset[1:] # 0-Frequenzen abschneiden
len_fft_dataset = len(fft_dataset)
print('number of data points after fft = '+ str(len_fft_dataset))
#fft_dataset[400:] = 0
print()





############################### Butter ##############################

# Filter
sample_rate = 5

def get_iir_coefficients(freq_pass, freq_stop):
    (order, wn) = sp.signal.buttord(freq_pass, freq_stop, 0.1, 20, analog=False, fs=sample_rate)
    print('order and wn:')
    print(order, wn)
    butter = sp.signal.butter(order, wn, "lowpass", fs=sample_rate)
    # print(butter)
    return butter

iir_coefficients_b, iir_coefficients_a = get_iir_coefficients(0.5, 4)
y = sp.signal.lfilter(iir_coefficients_b, iir_coefficients_a, locdataset)

w, h = sp.signal.freqz(iir_coefficients_b, iir_coefficients_a, fs=sample_rate)


# Plot
nrow = 4
ncol = 1

fig1, ax1 = plt.subplots(nrow, ncol, sharex=False, sharey=False, figsize=(8, 10), dpi=100)

ax1[0].plot(locdataset, "green", label='data')
ax1[1].plot(fft_dataset, "red", label='fft data')
ax1[2].plot(y, "blue", label='filtered (Butterworth)')
ax1[3].plot(w, h, "orange", label='frequency response')

ax1[0].set_xlabel("Time [s]", fontsize=14)
ax1[0].set_ylabel("Temperature [°C]", fontsize=14)
ax1[0].set_xlim(-10,2750)
ax1[0].set_ylim(30,38)
# ax1[0].tick_params(axis="y")
ax1[0].set_title("Data", fontsize=20)

ax1[1].set_xlabel("Frequency [1/s]", fontsize=14)
ax1[1].set_ylabel("Amplitude", fontsize=14)
ax1[1].set_xlim(-10,2750)
# ax1[1].tick_params(axis="y")
ax1[1].set_title("FFT Data", fontsize=20)

ax1[2].set_xlabel("Time [s]", fontsize=14)
ax1[2].set_ylabel("Temperature [°C]", fontsize=14)
ax1[2].set_xlim(-10,2750)
ax1[2].set_ylim(30,38)
# ax1[2].tick_params(axis="y")
ax1[2].set_title("Filtered Data (Butterworth)", fontsize=20)

ax1[3].set_xlabel("Frequency [1/s]", fontsize=14)
ax1[3].set_ylabel("Gain", fontsize=14)
# ax1[3].tick_params(axis="y")
ax1[3].set_title("Frequency Response (Butterworth)", fontsize=20)

ax1[0].grid()
ax1[1].grid()
ax1[2].grid()
ax1[3].grid()
# ax1[0].legend(fontsize = 15)
# ax1[1].legend(fontsize = 15)
# ax1[2].legend(fontsize = 15)
# ax1[3].legend(fontsize = 15)

# fig1.suptitle("third try wtf", fontsize=20)
# fig1.autofmt_xdate()
fig1.tight_layout(h_pad=2)

"""
print(len(locdataset))

fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=100)

ax2.plot(locdataset, "green", label='data', linewidth=1)
ax2.plot(y, "blue", label='filtered data', linewidth=2)

ax2.set_xlabel("Time [s]s", fontsize=14)
ax2.set_ylabel("Temperature [°C]", fontsize=14)
ax2.set_title("Comparison of Data with Filtered Data (Butterworth)", fontsize=20)

ax2.grid()
ax2.legend(fontsize = 15)

# fig2.suptitle("Butter Filter", fontsize=20)
fig2.tight_layout(h_pad=2)

"""

############################### Ellip ##############################

# Filter
sample_rate = 5

def get_iir_coefficients(freq_pass, freq_stop):
    (order, wn) = sp.signal.ellipord(freq_pass, freq_stop, 0.1, 20, analog=False, fs=sample_rate)
    print('order and wn:')
    print(order, wn)
    ellip = sp.signal.ellip(order, .1, 20, wn, "lowpass", output='ba' ,fs=sample_rate)
    # print(ellip)
    return ellip

iir_coefficients_b, iir_coefficients_a = get_iir_coefficients(0.5, 4)
y = sp.signal.lfilter(iir_coefficients_b, iir_coefficients_a, locdataset)

w, h = sp.signal.freqz(iir_coefficients_b, iir_coefficients_a, fs=sample_rate)


# Plot
nrow = 4
ncol = 1

fig3, ax3 = plt.subplots(nrow, ncol, sharex=False, sharey=False, figsize=(8, 10), dpi=100)

ax3[0].plot(locdataset, "green", label='data')
ax3[1].plot(fft_dataset, "red", label='fft data')
ax3[2].plot(y, "blue", label='filtered (Butterworth)')
ax3[3].plot(w, h, "orange", label='frequency response')

ax3[0].set_xlabel("Time [s]", fontsize=14)
ax3[0].set_ylabel("Temperature [°C]", fontsize=14)
ax3[0].set_xlim(-10,2750)
ax3[0].set_ylim(30,38)
ax3[0].set_title("Data", fontsize=20)

ax3[1].set_xlabel("Frequency [1/s]", fontsize=14)
ax3[1].set_ylabel("Amplitude", fontsize=14)
ax3[1].set_xlim(-10,2750)
ax3[1].set_title("fft Data", fontsize=20)

ax3[2].set_xlabel("Time [s]", fontsize=14)
ax3[2].set_ylabel("Temperature [°C]", fontsize=14)
ax3[2].set_xlim(-10,2750)
ax3[2].set_ylim(30,38)
ax3[2].set_title("filtered Data (Elliptic)", fontsize=20)

ax3[3].set_xlabel("Frequency [1/s]", fontsize=14)
ax3[3].set_ylabel("Gain", fontsize=14)
ax3[3].set_title("frequency response (Elliptic)", fontsize=20)

ax3[0].grid()
ax3[1].grid()
ax3[2].grid()
ax3[3].grid()

fig3.tight_layout(h_pad=2)


"""fig4, ax4 = plt.subplots(figsize=(10, 6), dpi=100)

ax4.plot(locdataset, "green", label='data', linewidth=1)
ax4.plot(y, "blue", label='filtered data', linewidth=2)

ax4.set_xlabel("Time [s]", fontsize=14)
ax4.set_ylabel("Temperature [°C]", fontsize=14)
ax4.set_title("Comparison of Data with Filtered Data (Elliptic)", fontsize=20)

ax4.grid()
ax4.legend(fontsize = 15)

fig4.tight_layout(h_pad=2)"""


"""response = 20 * np.log10(np.abs(fftshift(delta_Temp189 / abs(delta_Temp189).max())))"""
#plt.plot(response)
#plt.plot(FT_Temp040)
#plt.title("Frequency response of the boxcar window")
#plt.ylabel("Normalized magnitude [dB]")
#plt.xlabel("Normalized frequency [cycles per sample]")



plt.show()
