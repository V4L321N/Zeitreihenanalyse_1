# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 15:09:50 2022

@author: flori
"""

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
"""plt.plot(locTemp040, label = 'temp040')"""

dataset066 = pd.DataFrame(data=pd.read_csv("Rec066.csv"))
locTemp066 = dataset066.loc[:, 'Local Temp [degC]']
xPos066 = dataset066.loc[:, 'Position x [cm]']
yPos066 = dataset066.loc[:, 'Position y [cm]']
distOpt066 = dataset066.loc[:, 'Dist Opt [cm]']
"""plt.plot(locTemp066, '--', label = 'temp066')"""

dataset096 = pd.DataFrame(data=pd.read_csv("Rec096.csv"))
locTemp096 = dataset096.loc[:, 'Local Temp [degC]']
xPos096 = dataset096.loc[:, 'Position x [cm]']
yPos096 = dataset096.loc[:, 'Position y [cm]']
distOpt096 = dataset096.loc[:, 'Dist Opt [cm]']
"""plt.plot(locTemp096, label = 'temp096')"""

dataset189 = pd.DataFrame(data=pd.read_csv("Rec189.csv"))
locTemp189 = dataset189.loc[:, 'Local Temp [degC]']
xPos189 = dataset189.loc[:, 'Position x [cm]']
yPos189 = dataset189.loc[:, 'Position y [cm]']
distOpt189 = dataset189.loc[:, 'Dist Opt [cm]']
"""plt.plot(locTemp189, label = 'temp189')"""

dataset213 = pd.DataFrame(data=pd.read_csv("Rec213.csv"))
locTemp213 = dataset213.loc[:, 'Local Temp [degC]']
xPos213 = dataset213.loc[:, 'Position x [cm]']
yPos213 = dataset213.loc[:, 'Position y [cm]']
distOpt213 = dataset213.loc[:, 'Dist Opt [cm]']
"""plt.plot(locTemp213, '--', label = 'temp213')
plt.xlabel('time [s]', fontsize = 15)
plt.ylabel('local Temperature [°C]', fontsize = 15)
plt.title('trajectories (all in one)', fontsize = 20)
plt.legend()
plt.show()"""



#plot 2
nrow = 5
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

fig2.supxlabel('time [s]', fontsize = 15)
fig2.supylabel('local Temperature [°C]', fontsize = 15)

fig2.suptitle('trajectories (subplots)', fontsize = 20)
plt.subplots_adjust(top=1)
fig2.tight_layout(h_pad=2)
plt.show()


"""x,y = X_remove_NaN(IB_1),Y_remove_NaN(IB_1)
IB1.plot(ArenaX, ArenaY, color="black", linestyle="dashed")
IB1.plot(GradientX, GradientY, color="black", linestyle="dotted", dash_capstyle='round')
IB1.set_xticks([])
IB1.set_yticks([])
R = 0.0
G = 0.0
B = 1.0
time = len(x)
for i in range(time-1):
    IB1.plot((x[i],x[i+1]),(y[i],y[i+1]),color=(R,G,B))
    R+=(1/time)
    #G+=0.001
    B-=(1/time)"""

