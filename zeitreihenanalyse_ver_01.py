import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

"""fig1, ((Rec040, Rec066, Rec096, Rec189, Rec213)) = plt.subplots(1, 5, figsize=(18,3))

R = 0.0
G = 0.0
B = 1.0
Rec040.set_xlim(-2,62)
Rec040.set_ylim(-2,62)
Rec040.set_yticks([0, 20, 40, 60])
Rec040.set_ylabel("y")
Rec040.set_xlabel("x")
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
time = len(xPos213)
for i in range(1,time-1):
    Rec213.plot((xPos213.iloc[i], xPos213.iloc[i+1]), (yPos213.iloc[i], yPos213.iloc[i+1]), color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)
plt.show()"""

"""nrow = 5
ncol = 1
fig2, ax2 = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(9, 6), dpi=100)
ax2[0].plot(locTemp040)
ax2[0].grid()
ax2[0].annotate("run 040", (2650,27))
#ax2[0].set_title('temp040')
ax2[1].plot(locTemp066)
ax2[1].grid()
ax2[1].annotate("run 066", (2650,27))
#ax2[1].set_title('temp066')
ax2[2].plot(locTemp096)
ax2[2].grid()
ax2[2].annotate("run 096", (2650,27))
#ax2[2].set_title('temp096')
ax2[3].plot(locTemp189)
ax2[3].grid()
ax2[3].annotate("run 189", (2650,27))
#ax2[3].set_title('temp189')
ax2[4].plot(locTemp213)
ax2[4].grid()
ax2[4].annotate("run 213", (2650,27))
#ax2[4].set_title('temp213')


#plt.xlabel('time [s]')
#plt.ylabel('local Temperature [°C]')
#plt.show()


fig2.supxlabel('time [s]', fontsize = 15)
fig2.supylabel('local Temperature [°C]', fontsize = 15)

fig2.suptitle('trajectories (subplots)', fontsize = 20)
plt.subplots_adjust(top=1)
fig2.tight_layout(h_pad=2)
plt.show()"""

"""nrow = 5
ncol = 1
fig3, ax3 = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(9, 6), dpi=100)
ax3[0].plot(vel040)
ax3[0].grid()
ax3[0].annotate("run 040", (2650,27))
ax3[1].plot(vel066)
ax3[1].grid()
ax3[1].annotate("run 066", (2650,27))
ax3[2].plot(vel096)
ax3[2].grid()
ax3[2].annotate("run 096", (2650,27))
ax3[3].plot(vel189)
ax3[3].grid()
ax3[3].annotate("run 189", (2650,27))
ax3[4].plot(vel213)
ax3[4].grid()
ax3[4].annotate("run 213", (2650,27))

fig3.supxlabel('time [s]', fontsize = 15)
fig3.supylabel('velocity [cm/s]', fontsize = 15)

fig3.suptitle('velocities (subplots)', fontsize = 20)
plt.subplots_adjust(top=1)
fig3.tight_layout(h_pad=2)
plt.show()"""

nrow = 5
ncol = 1
fig4, ax4 = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(9, 6), dpi=100)
ax4[0].plot(velFFT040)
ax4[0].grid()
ax4[0].annotate("run 040", (2650,27))
ax4[1].plot(velFFT066)
ax4[1].grid()
ax4[1].annotate("run 066", (2650,27))
ax4[2].plot(velFFT096)
ax4[2].grid()
ax4[2].annotate("run 096", (2650,27))
ax4[3].plot(velFFT189)
ax4[3].grid()
ax4[3].annotate("run 189", (2650,27))
ax4[4].plot(velFFT213)
ax4[4].grid()
ax4[4].annotate("run 213", (2650,27))

fig4.supxlabel('frequency [1/s]', fontsize = 15)
fig4.supylabel('amplitude', fontsize = 15)

fig4.suptitle('FFT velocities (subplots)', fontsize = 20)
plt.subplots_adjust(top=1)
fig4.tight_layout(h_pad=2)
plt.show()