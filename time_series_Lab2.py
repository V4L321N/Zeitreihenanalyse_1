import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.fftpack import fftshift
import scipy.signal as signal

# begin: read dataset and initialize lists ---------------------------------------------------------------------------------------
head = list(pd.DataFrame(data=pd.read_csv(r"C:\Users\vstok\Desktop\zeitreihenanalyse_1-main\Rec040.csv")))
# ['Frame No.', 'Time ', 'Local Temp [degC]', 'Position x [cm]', 'Position y [cm]', 'Dist Opt [cm]', 'Dist Opt Avg [cm]', 'T Opt [degC]']

dataset040 = pd.DataFrame(data=pd.read_csv(r"C:\Users\vstok\Desktop\zeitreihenanalyse_1-main\Rec040.csv"))
locTemp040 = dataset040.loc[:, 'Local Temp [degC]']
xPos040 = dataset040.loc[:, 'Position x [cm]']
yPos040 = dataset040.loc[:, 'Position y [cm]']
distOpt040 = dataset040.loc[:, 'Dist Opt [cm]']

dataset066 = pd.DataFrame(data=pd.read_csv(r"C:\Users\vstok\Desktop\zeitreihenanalyse_1-main\Rec066.csv"))
locTemp066 = dataset066.loc[:, 'Local Temp [degC]']
xPos066 = dataset066.loc[:, 'Position x [cm]']
yPos066 = dataset066.loc[:, 'Position y [cm]']
distOpt066 = dataset066.loc[:, 'Dist Opt [cm]']

dataset096 = pd.DataFrame(data=pd.read_csv(r"C:\Users\vstok\Desktop\zeitreihenanalyse_1-main\Rec096.csv"))
locTemp096 = dataset096.loc[:, 'Local Temp [degC]']
xPos096 = dataset096.loc[:, 'Position x [cm]']
yPos096 = dataset096.loc[:, 'Position y [cm]']
distOpt096 = dataset096.loc[:, 'Dist Opt [cm]']

dataset189 = pd.DataFrame(data=pd.read_csv(r"C:\Users\vstok\Desktop\zeitreihenanalyse_1-main\Rec189.csv"))
locTemp189 = dataset189.loc[:, 'Local Temp [degC]']
xPos189 = dataset189.loc[:, 'Position x [cm]']
yPos189 = dataset189.loc[:, 'Position y [cm]']
distOpt189 = dataset189.loc[:, 'Dist Opt [cm]']

dataset213 = pd.DataFrame(data=pd.read_csv(r"C:\Users\vstok\Desktop\zeitreihenanalyse_1-main\Rec213.csv"))
locTemp213 = dataset213.loc[:, 'Local Temp [degC]']
xPos213 = dataset213.loc[:, 'Position x [cm]']
yPos213 = dataset213.loc[:, 'Position y [cm]']
distOpt213 = dataset213.loc[:, 'Dist Opt [cm]']
# end: read dataset and initialize lists ---------------------------------------------------------------------------------------



def function_FFT(item):
    return np.fft.fft(item, norm="ortho")

FT_Temp040 = function_FFT(locTemp040)
FT_Temp066 = function_FFT(locTemp066)
FT_Temp096 = function_FFT(locTemp096)
FT_Temp189 = function_FFT(locTemp189)
FT_Temp213 = function_FFT(locTemp213)

delta_Temp040 = np.fft.ifft(FT_Temp040)
delta_Temp066 = np.fft.ifft(FT_Temp066)
delta_Temp096 = np.fft.ifft(FT_Temp096)
delta_Temp189 = np.fft.ifft(FT_Temp189)
delta_Temp213 = np.fft.ifft(FT_Temp213)

#plt.plot(delta_Temp040)
#plt.plot(delta_Temp066)
#plt.plot(delta_Temp096)
#plt.plot(delta_Temp189)
#plt.plot(delta_Temp213)


"""# begin: Filter BUTTER ------------------------------------------------------------------------------------------
but_sample_rate = 25
def butter_iir_coefficients(freq_pass, freq_stop):
    (order, wn) = sp.signal.buttord(freq_pass, freq_stop, 0.1, 20, analog=False, fs=but_sample_rate)
    print('order and wn:')
    print(order, wn)
    butter = sp.signal.butter(order, wn, "lowpass", fs=but_sample_rate)
    # print(butter)
    return butter

butter_iir_coefficients_b, butter_iir_coefficients_a = butter_iir_coefficients(0.5, 4)
but_y = sp.signal.lfilter(butter_iir_coefficients_b, butter_iir_coefficients_a, locTemp189)
but_w, but_h = sp.signal.freqz(butter_iir_coefficients_b, butter_iir_coefficients_a, fs=but_sample_rate)

plt.plot(but_h)
plt.show()

plt.figure(figsize=(12,4))
plt.xlabel('t [s]')
plt.ylabel('T [°C]')
plt.plot(but_y, label='butterworth filtered')
plt.plot(locTemp189, color='black', linestyle='solid', alpha=0.5, label='original data')
plt.tight_layout()
plt.legend()
plt.show()
# end: Filter BUTTER ---------------------------------------------------------------------------------------------

# begin: Filter ELLIPTIC ---------------------------------------------------------------------------------------------
ell_sample_rate = 25

def elliptic_iir_coefficients(freq_pass, freq_stop):
    (order, wn) = sp.signal.ellipord(freq_pass, freq_stop, 0.1, 20, analog=False, fs=ell_sample_rate)
    print('order and wn:')
    print(order, wn)
    ellip = sp.signal.ellip(order, .1, 20, wn, "lowpass", output='ba' ,fs=ell_sample_rate)
    # print(ellip)
    return ellip

elliptic_iir_coefficients_b, elliptic_iir_coefficients_a = elliptic_iir_coefficients(0.5, 4)
ell_y = sp.signal.lfilter(elliptic_iir_coefficients_b, elliptic_iir_coefficients_a, locTemp189)

ell_w, ell_h = sp.signal.freqz(elliptic_iir_coefficients_b, elliptic_iir_coefficients_a, fs=ell_sample_rate)

plt.plot(ell_h)
plt.show()

plt.figure(figsize=(12,4))
plt.xlabel('t [s]')
plt.ylabel('T [°C]')
plt.plot(ell_y, label='elliptic filtered')
plt.plot(locTemp189, color='black', linestyle='solid', alpha=0.5, label='original data')
plt.tight_layout()
plt.legend()
plt.show()
# end: Filter ELLIPTIC ---------------------------------------------------------------------------------------------


# begin: Filter BOXCAR ------------------------------------------------------------------------------------------
def moving_average(time_series, window):
    moving_avg = np.convolve(time_series, np.ones(window)/window, mode='valid')
    return moving_avg
windowBOX = 25
BOX_filtered_time_series = moving_average(locTemp189, windowBOX)

plt.figure(figsize=(12,4))
plt.xlabel('t [s]')
plt.ylabel('T [°C]')
plt.plot(BOX_filtered_time_series, label='boxcar filtered')
plt.plot(locTemp189, color='black', linestyle='solid', alpha=0.5, label='original data')
plt.tight_layout()
plt.legend()
plt.show()
# end: Filter BOXCAR ---------------------------------------------------------------------------------------------

# begin: Filter TRIANGLE ------------------------------------------------------------------------------------------
def triangle_filter(time_series, window):
    triangle = np.concatenate((np.arange(1, window+1), np.arange(window-1, 0, -1))) # create triangle window
    plt.plot(triangle)
    plt.show()
    triangle = triangle / triangle.sum() # normalize window
    filtered_time_series = np.convolve(time_series, triangle, mode='same')
    return filtered_time_series
windowTRI = 25
TRI_filtered_time_series = triangle_filter(locTemp189, windowTRI)

plt.figure(figsize=(12,4))
plt.xlabel('t [s]')
plt.ylabel('T [°C]')
plt.plot(TRI_filtered_time_series, label='triangle filtered')
plt.plot(locTemp189, color='black', linestyle='solid', alpha=0.5, label='original data')
plt.tight_layout()
plt.legend()
plt.show()
# end: Filter TRIANGLE ---------------------------------------------------------------------------------------------

# begin: Filter HANNING ------------------------------------------------------------------------------------------
def hanning_filter(time_series, window):
    hanning = np.hanning(window) # create hanning window
    filtered_time_series = np.convolve(time_series, hanning, mode='same') / sum(hanning)
    return filtered_time_series

window = 25
HANN_filtered_time_series = hanning_filter(locTemp189, window)

plt.figure(figsize=(12,4))
plt.xlabel('t [s]')
plt.ylabel('T [°C]')
plt.plot(HANN_filtered_time_series, label='hanning filtered')
plt.plot(locTemp189, color='black', linestyle='solid', alpha=0.5, label='original data')
plt.tight_layout()
plt.legend()
plt.show()
# end: Filter HANNING --------------------------------------------------------------------------------------------"""

# begin: Filter Chebyshev ------------------------------------------------------------------------------------------
def chebyshev_filter(time_series, cutoff, fs, order, ripple):
    b, a = signal.cheby1(order, ripple, cutoff / (0.5 * fs), btype='low', analog=False, output='ba')
    filtered_time_series = signal.filtfilt(b, a, time_series)
    return filtered_time_series

cutoff = 0.35
fs = 50
order = 1
ripple = 1
CHEBY_filtered_time_series = chebyshev_filter(locTemp189, cutoff, fs, order, ripple)

plt.figure(figsize=(12,4))
plt.xlabel('t [s]')
plt.ylabel('T [°C]')
plt.plot(CHEBY_filtered_time_series, label='chebyshev filtered')
plt.plot(locTemp189, color='black', linestyle='solid', alpha=0.5, label='original data')
plt.tight_layout()
plt.legend()
plt.show()
# end: Filter Chebyshev ------------------------------------------------------------------------------------------


