import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy


# begin: read dataset and initialize lists ---------------------------------------------------------------------------------------

head = list(pd.DataFrame(data=pd.read_csv("Rec040.csv")))
# ['Frame No.', 'Time ', 'Local Temp [degC]', 'Position x [cm]', 'Position y [cm]', 'Dist Opt [cm]', 'Dist Opt Avg [cm]', 'T Opt [degC]']

dataset040 = pd.DataFrame(data=pd.read_csv("Rec040.csv"))
locTemp040 = dataset040.loc[:, 'Local Temp [degC]']
xPos040 = dataset040.loc[:, 'Position x [cm]']
yPos040 = dataset040.loc[:, 'Position y [cm]']
distOpt040 = dataset040.loc[:, 'Dist Opt [cm]']

dataset066 = pd.DataFrame(data=pd.read_csv("Rec066.csv"))
locTemp066 = dataset066.loc[:, 'Local Temp [degC]']
xPos066 = dataset066.loc[:, 'Position x [cm]']
yPos066 = dataset066.loc[:, 'Position y [cm]']
distOpt066 = dataset066.loc[:, 'Dist Opt [cm]']

dataset096 = pd.DataFrame(data=pd.read_csv("Rec096.csv"))
locTemp096 = dataset096.loc[:, 'Local Temp [degC]']
xPos096 = dataset096.loc[:, 'Position x [cm]']
yPos096 = dataset096.loc[:, 'Position y [cm]']
distOpt096 = dataset096.loc[:, 'Dist Opt [cm]']

dataset189 = pd.DataFrame(data=pd.read_csv("Rec189.csv"))
locTemp189 = dataset189.loc[:, 'Local Temp [degC]']
xPos189 = dataset189.loc[:, 'Position x [cm]']
yPos189 = dataset189.loc[:, 'Position y [cm]']
distOpt189 = dataset189.loc[:, 'Dist Opt [cm]']

dataset213 = pd.DataFrame(data=pd.read_csv("Rec213.csv"))
locTemp213 = dataset213.loc[:, 'Local Temp [degC]']
xPos213 = dataset213.loc[:, 'Position x [cm]']
yPos213 = dataset213.loc[:, 'Position y [cm]']
distOpt213 = dataset213.loc[:, 'Dist Opt [cm]']

# end: read dataset and initialize lists ---------------------------------------------------------------------------------------

# begin: define needed functions ---------------------------------------------------------------------------------------

def calc_vel(x, y):
    list_vel = []
    xPos = x
    yPos = y
    for t in range(len(x) - 1):
        velocity = np.sqrt((x.iloc[t + 1] - x.iloc[t]) ** 2 + (y.iloc[t + 1] - y.iloc[t]) ** 2)
        list_vel.append(velocity)
    return list_vel

vel040 = calc_vel(xPos040,yPos040)
vel066 = calc_vel(xPos066,yPos066)
vel096 = calc_vel(xPos096,yPos096)
vel189 = calc_vel(xPos189,yPos189)
vel213 = calc_vel(xPos213,yPos213)

def velocity_FFT(item):
    return np.fft.fft(item, norm="ortho")

velFFT040 = velocity_FFT(vel040)
velFFT066 = velocity_FFT(vel066)
velFFT096 = velocity_FFT(vel096)
velFFT189 = velocity_FFT(vel189)
velFFT213 = velocity_FFT(vel213)

def temperature_FFT(item):
    return np.fft.fft(item, norm="ortho")

tempFFT040 = temperature_FFT(locTemp040)
tempFFT066 = temperature_FFT(locTemp066)
tempFFT096 = temperature_FFT(locTemp096)
tempFFT189 = temperature_FFT(locTemp189)
tempFFT213 = temperature_FFT(locTemp213)

def is_it_walking(item):
    ZerosOnes = []
    for i in range(len(item)):
        if item[i] < 0.25:
            ZerosOnes.append(0)
        else:
            ZerosOnes.append(1)
    return ZerosOnes

ZeroOne040 = is_it_walking(vel040)
ZeroOne066 = is_it_walking(vel066)
ZeroOne096 = is_it_walking(vel096)
ZeroOne189 = is_it_walking(vel189)
ZeroOne213 = is_it_walking(vel213)

def set_cut(item):
    cut_list = []
    inverse_cut_list = []
    truncated_list = []
    for i in range(len(item)):
        if item[i] < 0.15:
            cut_list.append(i)
        else:
            inverse_cut_list.append(i)
    return cut_list, inverse_cut_list

cuts040 = set_cut(vel040)
cuts066 = set_cut(vel066)
cuts096 = set_cut(vel096)
cuts189 = set_cut(vel189)
cuts213 = set_cut(vel213)

#print(cuts040[1])

def timeblocks(item):
    init_list = []
    blocked_df = []
    for i in range(len(item)-1):
        if item[i+1] - item[i] == 1:
            init_list.append(i)
        else:
            init_list.append(i)
            blocked_df.append(init_list)
            init_list = []
    return blocked_df

# end: define needed functions ---------------------------------------------------------------------------------------

# begin analyse run040 from 806 to 2150 ----VERY MUCH WORK IN PROGRESS----------------------------------------------------------------------------------------

"""#print((cuts040[0])[228], (cuts040[0])[1579])       
detail040 = locTemp040[(cuts040[0])[227]:(cuts040[0])[1579]]
#print(detail040)
detailFFT040 = temperature_FFT(detail040)"""

"""plt.plot(detail040)
plt.show()
plt.plot(detailFFT040[1:])
#plt.xscale("log")
#plt.yscale("log")
plt.show()
plt.phase_spectrum(detail040)    #), Fs=None, Fc=None, window=None, pad_to=None, sides=None, *, data=None, **kwargs)
plt.show()
#plt.magnitude_spectrum(detail040) #), Fs=None, Fc=None, window=None, pad_to=None, sides=None, scale=None, *, data=None, **kwargs)[source]
#plt.xscale("log")
#plt.yscale("log")
#plt.show()"""


"""print(np.abs(tempFFT040[0]))
print(np.abs(detailFFT040[0]))
subtract040 = []
print(len(detailFFT040))
for i in range(1, len(detailFFT040)):
    subtract040.append(np.abs(tempFFT040[i]) - np.abs(detailFFT040[i]))
plt.plot(np.abs(tempFFT040[1:]), label='1', alpha=0.5)
plt.plot(np.abs(detailFFT040[1:]), label='-2', alpha=0.5)
#plt.plot(np.abs(subtract040[1:]), label='=3', alpha=0.5)
plt.legend()
#plt.xscale("log")
#plt.yscale("log")
plt.show()"""

"""subtract040 = []
for i in range(int(len(detailFFT040)/2)):
    subtract = tempFFT040[i] - 2*detailFFT040[i]
    subtract040.append(subtract)
for i in range(int(len(detailFFT040)/2+1), int(len(tempFFT040))):
    rest = tempFFT040[i]
    subtract040.append(rest)"""

"""#cut = tempFFT040[1365:2738]
plt.figure(figsize=(9,6))
plt.plot(tempFFT040[1:int(len(detailFFT040)/2)], linewidth=3, label="original")
plt.plot(subtract040[1:int(len(detailFFT040)/2)], linewidth=1, color="red", label="final")
plt.plot(detailFFT040[1:int(len(detailFFT040)/2)], linewidth=1, linestyle="dashed", color="pink", label="fluctuation")

print(len(subtract040))
print(len(tempFFT040))
print(len(detailFFT040))
plt.xlabel("frequency [1/s]")
plt.ylabel("Amplitude")
plt.legend()
plt.show()"""

"""plt.figure(figsize=(11,3))
inverseFFT1 = np.fft.ifft(subtract040)
inverseFFT2 = np.fft.ifft(tempFFT040)
difference = inverseFFT1[0] - inverseFFT2[0] 
plt.plot(inverseFFT2, linewidth=1, label="ifft(original)")
plt.plot(inverseFFT1-difference, linewidth=1, label="ifft(final)", color="red")
#plt.show()

#plt.plot(locTemp040/50, linewidth=1, color="blue", label="original data")
plt.xlabel("time [s]")
plt.ylabel("temp. metric [°C]")
plt.legend()
plt.show()"""

# end analyse run040 from 806 to 2150 ------------------------------------------------------------------------------------------------------------------

# begin PLOT: red to blue trajectories --------------------------------------------------------------------------------
"""
fig1, ((Rec040, Rec066, Rec096, Rec189, Rec213)) = plt.subplots(1, 5, figsize=(18,3.6))
fig1.tight_layout(h_pad=2)
fig1.subplots_adjust(bottom=0.15, left=0.05)

R = 0.0
G = 0.0
B = 1.0
Rec040.set_xlim(-2,62)
Rec040.set_ylim(-2,62)
Rec040.set_yticks([0, 20, 40, 60])
Rec040.set_ylabel("y")
Rec040.set_xlabel("x")
Rec040.annotate("run 040", (45,0))
time = len(xPos040)
for i in range(1,time-1):
    Rec040.plot((xPos040.iloc[i], xPos040.iloc[i+1]), (yPos040.iloc[i], yPos040.iloc[i+1]), color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)

R = 0.0
G = 0.0
B = 1.0
Rec066.set_xlim(-2,62)
Rec066.set_ylim(-2,62)
Rec066.set_yticks([])
Rec066.set_xlabel("x")
Rec066.annotate("run 066", (45,0))
time = len(xPos066)
for i in range(1,time-1):
    Rec066.plot((xPos066.iloc[i], xPos066.iloc[i+1]), (yPos066.iloc[i], yPos066.iloc[i+1]), color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)

R = 0.0
G = 0.0
B = 1.0
Rec096.set_xlim(-2,62)
Rec096.set_ylim(-2,62)
Rec096.set_yticks([])
Rec096.set_xlabel("x")
Rec096.annotate("run 096", (45,0))
time = len(xPos096)
for i in range(1,time-1):
    Rec096.plot((xPos096.iloc[i], xPos096.iloc[i+1]), (yPos096.iloc[i], yPos096.iloc[i+1]), color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)

R = 0.0
G = 0.0
B = 1.0
Rec189.set_xlim(-2,62)
Rec189.set_ylim(-2,62)
Rec189.set_yticks([])
Rec189.set_xlabel("x")
Rec189.annotate("run 189", (45,0))
time = len(xPos189)
for i in range(1,time-1):
    Rec189.plot((xPos189.iloc[i], xPos189.iloc[i+1]), (yPos189.iloc[i], yPos189.iloc[i+1]), color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)

R = 0.0
G = 0.0
B = 1.0
Rec213.set_xlim(-2,62)
Rec213.set_ylim(-2,62)
Rec213.set_yticks([])
Rec213.set_xlabel("x")
Rec213.annotate("run 213", (45,0))
time = len(xPos213)
for i in range(1,time-1):
    Rec213.plot((xPos213.iloc[i], xPos213.iloc[i+1]), (yPos213.iloc[i], yPos213.iloc[i+1]), color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)
plt.show()"""

# end PLOT: red to blue trajectories --------------------------------------------------------------------------------

# begin PLOT: local temperature over time --------------------------------------------------------------------------------

"""nrow = 5
ncol = 1
fig2, ax2 = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(9, 9), dpi=100)

x = np.arange(0, len(locTemp040))
z = np.polyfit(x, locTemp040, 1)
p = np.poly1d(z)
ax2[0].plot(x, p(x), color="red")

ax2[0].plot(locTemp040)
ax2[0].grid()
ax2[0].annotate("run 040", (2650,27))
#ax2[0].set_title('temp040')

x = np.arange(0, len(locTemp066))
z = np.polyfit(x, locTemp066, 1)
p = np.poly1d(z)
ax2[1].plot(x, p(x), color="red")

ax2[1].plot(locTemp066)
ax2[1].grid()
ax2[1].annotate("run 066", (2650,27))
#ax2[1].set_title('temp066')

x = np.arange(0, len(locTemp096))
z = np.polyfit(x, locTemp096, 1)
p = np.poly1d(z)
ax2[2].plot(x, p(x), color="red")

ax2[2].plot(locTemp096)
ax2[2].grid()
ax2[2].annotate("run 096", (2650,27))
#ax2[2].set_title('temp096')

x = np.arange(0, len(locTemp189))
z = np.polyfit(x, locTemp189, 1)
p = np.poly1d(z)
ax2[3].plot(x, p(x), color="red")

ax2[3].plot(locTemp189)
ax2[3].grid()
ax2[3].annotate("run 189", (2650,27))
#ax2[3].set_title('temp189')

x = np.arange(0, len(locTemp213))
z = np.polyfit(x, locTemp213, 1)
p = np.poly1d(z)
ax2[4].plot(x, p(x), color="red")

ax2[4].plot(locTemp213)
ax2[4].grid()
ax2[4].annotate("run 213", (2650,27))
#ax2[4].set_title('temp213')
fig2.supxlabel('time [s]', fontsize = 15)
fig2.supylabel('local Temperature [°C]', fontsize = 15)
plt.subplots_adjust(top=1)
fig2.tight_layout(h_pad=2)
plt.show()"""

# end PLOT: local temperature over time --------------------------------------------------------------------------------

# begin PLOT: CUT local temperature over time --------------------------------------------------------------------------------

"""nrow = 5
ncol = 1
fig2, ax2c = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(9, 9), dpi=100)
for x in range(len(ZeroOne040)-1):
    if ZeroOne040[x] == 0 and ZeroOne040[x+1] != 0:
        ax2c[0].axvline(x, color="red")
ax2c[0].plot(locTemp040)
ax2c[0].grid()
ax2c[0].annotate("run 040", (2650,27))
#ax2c[0].set_title('temp040')
for x in range(len(ZeroOne066)-1):
    if ZeroOne066[x] == 0 and ZeroOne066[x+1] != 0:
        ax2c[1].axvline(x, color="red")
ax2c[1].plot(locTemp066)
ax2c[1].grid()
ax2c[1].annotate("run 066", (2650,27))
#ax2c[1].set_title('temp066')
for x in range(len(ZeroOne096)-1):
    if ZeroOne096[x] == 0 and ZeroOne096[x+1] != 0:
        ax2c[2].axvline(x, color="red")
ax2c[2].plot(locTemp096)
ax2c[2].grid()
ax2c[2].annotate("run 096", (2650,27))
#ax2c[2].set_title('temp096')
for x in range(len(ZeroOne189)-1):
    if ZeroOne189[x] == 0 and ZeroOne189[x+1] != 0:
        ax2c[3].axvline(x, color="red")
ax2c[3].plot(locTemp189)
ax2c[3].grid()
ax2c[3].annotate("run 189", (2650,27))
#ax2c[3].set_title('temp189')
for x in range(len(ZeroOne213)-1):
    if ZeroOne213[x] == 0 and ZeroOne213[x+1] != 0:
        ax2c[4].axvline(x, color="red")
ax2c[4].plot(locTemp213)
ax2c[4].grid()
ax2c[4].annotate("run 213", (2650,27))
#ax2c[4].set_title('temp213')
fig2.supxlabel('time [s]', fontsize = 15)
fig2.supylabel('local Temperature [°C]', fontsize = 15)
plt.subplots_adjust(top=1)
fig2.tight_layout(h_pad=2)
plt.show()"""

# end PLOT: CUT local temperature over time --------------------------------------------------------------------------------

# begin PLOT: velocity over time --------------------------------------------------------------------------------

"""nrow = 5
ncol = 1
fig3, ax3 = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(9, 9), dpi=100)

ax3[0].plot(vel040)
ax3[0].grid()
ax3[0].annotate("run 040", (2650,0))
ax3[1].plot(vel066)
ax3[1].grid()
ax3[1].annotate("run 066", (2650,0))
ax3[2].plot(vel096)
ax3[2].grid()
ax3[2].annotate("run 096", (2650,0))
ax3[3].plot(vel189)
ax3[3].grid()
ax3[3].annotate("run 189", (2650,0))
ax3[4].plot(vel213)
ax3[4].grid()
ax3[4].annotate("run 213", (2650,0))

fig3.supxlabel('time [s]', fontsize = 15)
fig3.supylabel('velocity [cm/s]', fontsize = 15)
plt.subplots_adjust(top=1)
fig3.tight_layout(h_pad=2)
plt.show()"""

# end PLOT: velocity over time --------------------------------------------------------------------------------

# begin PLOT: CUT velocity over time --------------------------------------------------------------------------------

"""nrow = 5
ncol = 1
fig3, ax3c = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(9, 9), dpi=100)
for x in range(len(ZeroOne040)-1):
    if ZeroOne040[x] == 0 and ZeroOne040[x+1] != 0:
        ax3c[0].axvline(x, color="red")
ax3c[0].plot(vel040)
ax3c[0].grid()
ax3c[0].annotate("run 040", (2650,0))
for x in range(len(ZeroOne066)-1):
    if ZeroOne066[x] == 0 and ZeroOne066[x+1] != 0:
        ax3c[1].axvline(x, color="red")
ax3c[1].plot(vel066)
ax3c[1].grid()
ax3c[1].annotate("run 066", (2650,0))
for x in range(len(ZeroOne096)-1):
    if ZeroOne096[x] == 0 and ZeroOne096[x+1] != 0:
        ax3c[2].axvline(x, color="red")
ax3c[2].plot(vel096)
ax3c[2].grid()
ax3c[2].annotate("run 096", (2650,0))
for x in range(len(ZeroOne189)-1):
    if ZeroOne189[x] == 0 and ZeroOne189[x+1] != 0:
        ax3c[3].axvline(x, color="red")
ax3c[3].plot(vel189)
ax3c[3].grid()
ax3c[3].annotate("run 189", (2650,0))
for x in range(len(ZeroOne213)-1):
    if ZeroOne213[x] == 0 and ZeroOne213[x+1] != 0:
        ax3c[4].axvline(x, color="red")
ax3c[4].plot(vel213)
ax3c[4].grid()
ax3c[4].annotate("run 213", (2650,0))

fig3.supxlabel('time [s]', fontsize = 15)
fig3.supylabel('velocity [cm/s]', fontsize = 15)
plt.subplots_adjust(top=1)
fig3.tight_layout(h_pad=2)
plt.show()"""

# end PLOT: CUT velocity over time --------------------------------------------------------------------------------

# begin PLOT: FFTvelocity over frequency --------------------------------------------------------------------------------

"""nrow = 5
ncol = 1
fig4, ax4 = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(9, 9), dpi=100)
ax4[0].plot(abs(velFFT040[:round(len(velFFT040)/2)]))
ax4[0].grid()
ax4[0].annotate("run 040", (2650,27))
ax4[1].plot(abs(velFFT066[:round(len(velFFT066)/2)]))
ax4[1].grid()
ax4[1].annotate("run 066", (2650,27))
ax4[2].plot(abs(velFFT096[:round(len(velFFT096)/2)]))
ax4[2].grid()
ax4[2].annotate("run 096", (2650,27))
ax4[3].plot(abs(velFFT189[:round(len(velFFT189)/2)]))
ax4[3].grid()
ax4[3].annotate("run 189", (2650,27))
ax4[4].plot(abs(velFFT213[:round(len(velFFT213)/2)]))
ax4[4].grid()
ax4[4].annotate("run 213", (2650,27))

ax4[0].set_xscale("log")
ax4[0].set_yscale("log")
fig4.suptitle('FFT velocities', fontsize = 20)
fig4.supxlabel('frequency [1/s]', fontsize = 15)
fig4.supylabel('amplitude', fontsize = 15)
plt.subplots_adjust(top=1)
fig4.tight_layout(h_pad=2)
plt.show()"""

# end PLOT: FFTvelocity over frequency --------------------------------------------------------------------------------

# begin PLOT: FFTlocalTemp over frequency --------------------------------------------------------------------------------
"""
nrow = 5
ncol = 1
fig5, ax5 = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(9, 9), dpi=100)
ax5[0].plot(abs(tempFFT040[:round(len(velFFT040)/2)]))
ax5[0].grid()
ax5[0].annotate("run 040", (2650,27))
ax5[1].plot(abs(tempFFT066[:round(len(velFFT066)/2)]))
ax5[1].grid()
ax5[1].annotate("run 066", (2650,27))
ax5[2].plot(abs(tempFFT096[:round(len(velFFT096)/2)]))
ax5[2].grid()
ax5[2].annotate("run 096", (2650,27))
ax5[3].plot(abs(tempFFT189[:round(len(velFFT189)/2)]))
ax5[3].grid()
ax5[3].annotate("run 189", (2650,27))
ax5[4].plot(abs(tempFFT213[:round(len(velFFT213)/2)]))
ax5[4].grid()
ax5[4].annotate("run 213", (2650,27))
ax5[0].set_xscale("log")
ax5[0].set_yscale("log")
fig5.suptitle('FFT local temperature', fontsize = 20)
fig5.supxlabel('frequency [1/s]', fontsize = 15)
fig5.supylabel('amplitude', fontsize = 15)
plt.subplots_adjust(top=1)
fig5.tight_layout(h_pad=2)
plt.show()
"""

# end PLOT: FFTlocalTemp over frequency --------------------------------------------------------------------------------

# begin PLOT: CUT high freq and ifft -------------------------------------------------------------------------------------


nrow = 5
ncol = 1
fig6, ax6 = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(9, 9), dpi=100)
ax6[0].plot(np.fft.ifft(tempFFT040[:100]))
ax6[0].grid()
ax6[0].annotate("run 040", (2650,27))
ax6[1].plot(np.fft.ifft(tempFFT066[:100]))
ax6[1].grid()
ax6[1].annotate("run 066", (2650,27))
ax6[2].plot(np.fft.ifft(tempFFT096[:100]))
ax6[2].grid()
ax6[2].annotate("run 096", (2650,27))
ax6[3].plot(np.fft.ifft(tempFFT189[:100]))
ax6[3].grid()
ax6[3].annotate("run 189", (2650,27))
ax6[4].plot(np.fft.ifft(tempFFT213[:100]))
ax6[4].grid()
ax6[4].annotate("run 213", (2650,27))
fig6.suptitle('iFFT local temperature', fontsize = 20)
fig6.supxlabel('time [s]', fontsize = 15)
fig6.supylabel('temperature metric [°C]', fontsize = 15)
plt.subplots_adjust(top=1)
fig6.tight_layout(h_pad=2)
plt.show()

# end PLOT: CUT high freq and ifft ----------------------------------------------------------------------------------------

"""
def PSD(item):
    loop_L = round(len(item)/2)
    PSD_list = []
    for i in range(loop_L):
        C_2 = 1
        absolute = C_2 * (item[i] * np.conj(item[i]))
        PSD_list.append(absolute.real)
    return PSD_list

plt.plot(PSD(abs(velFFT040)))
plt.xscale('log')
plt.yscale('log')
plt.show()


end = round(len(velFFT040)/2)
plt.plot(abs(velFFT040[:end]))
plt.plot(abs(velFFT066[:end]))
plt.plot(abs(velFFT096[:end]))
plt.plot(abs(velFFT189[:end]))
plt.plot(abs(velFFT213[:end]))

plt.show()"""