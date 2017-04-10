import sys
import csv
import pprint
import matplotlib
import numpy  as np
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
plt.style.use('/home/rosh/Documents/EngD/Work/VidCodecWork/VideoProfilingData/analysis_tools/bmh_rosh.mplstyle')


def get_xy(data):
    x = [d[0] for d in data]
    y = [d[1] for d in data]
    return (x,y)

#############################################################################
#### scenario 1 : when CPU freq goes high because mem gov went low
#############################################################################

fig, axs = plt.subplots(4,1, figsize=(4, 8), sharex=True)
fig.canvas.set_window_title("scenario1")
axs = axs.ravel()


data_cpuutil = [(0,50), (1,50), (1,55), (2,55), (2,45), (3,45), (3,50), (5.25,50), (5.25,70), (5.50, 70), (5.50, 60), (7, 60), (7,40), (7.5,40), (7.5,50), (9,50)]
x,y = get_xy(data_cpuutil)
axs[0].plot(x,y)
axs[0].axvline(x=5.25, ls='--',lw=0.8, c='r', alpha=1.0)
axs[0].set_ylim([30, 80])
axs[0].set_ylabel("cpu_util")
axs[0].tick_params(axis='both', labelsize=11)



data_memutil = [(0,50), (1,50), (1,55), (2,55), (2,50), (3,50), (3,55), (5,55), (5,30), (5.25,30), (5.25,45), (7,45), (7,60), (7.25,60), (7.25,50), (9,50)]
x,y = get_xy(data_memutil)
axs[1].plot(x,y)
axs[1].set_ylim([20, 70])
axs[1].set_ylabel("mem_util")
axs[1].tick_params(axis='both', labelsize=11)

data_memfreq = [(0,3), (5.25,3), (5.25,1), (7.25,1), (7.25,3), (9,3)]
x,y = get_xy(data_memfreq)
axs[2].plot(x,y)
axs[2].set_ylim([0, 5])
axs[2].set_ylabel("mem_freq")
axs[2].tick_params(axis='both', labelsize=11)

data_cpufreq = [(0,3), (5.50,3), (5.50,4), (7.5,4), (7.5,3), (9,3)]
x,y = get_xy(data_cpufreq)
axs[3].plot(x,y)
axs[3].set_ylim([0, 5])
axs[3].set_ylabel("cpu_freq")
axs[3].tick_params(axis='both', labelsize=11)

for each_ax in axs:
    each_ax.axvline(x=5, ls='--',lw=0.8, c='grey', alpha=1.0)
    each_ax.axvline(x=5.25, ls='--',lw=0.8, c='grey', alpha=1.0)
    each_ax.axvline(x=5.5, ls='--',lw=0.8, c='grey', alpha=1.0)

    each_ax.axvline(x=7, ls='--',lw=0.8, c='grey', alpha=1.0)
    each_ax.axvline(x=7.25, ls='--',lw=0.8, c='grey', alpha=1.0)
    each_ax.axvline(x=7.5, ls='--',lw=0.8, c='grey', alpha=1.0)

plt.subplots_adjust(top=0.97, left=0.08, right=0.98, bottom=0.03, wspace=0.20, hspace=0.20)



#############################################################################
#### scenario 2 : when CPU freq does not change because mem gov went low
#############################################################################

fig, axs = plt.subplots(4,1, figsize=(4, 8), sharex=True)
fig.canvas.set_window_title("scenario2")
axs = axs.ravel()


data_cpuutil = [(0,50), (1,50), (1,55), (2,55), (2,45), (3,45), (3,50), (5.25,50), (5.25,55), (5.50, 55), (5.50, 60), (7, 60), (7,55), (7.25,55), (7.25,50), (9,50)]
x,y = get_xy(data_cpuutil)
axs[0].plot(x,y)
axs[0].axvline(x=5.25, ls='--',lw=0.8, c='r', alpha=1.0)
axs[0].set_ylim([30, 80])
axs[0].set_ylabel("cpu_util")
axs[0].tick_params(axis='both', labelsize=11)

data_memutil = [(0,50), (1,50), (1,55), (2,55), (2,50), (3,50), (3,55), (5,55), (5,30), (5.25,30), (5.25,45), (7,45), (7,60), (7.25,60), (7.25,50), (9,50)]
x,y = get_xy(data_memutil)
axs[1].plot(x,y)
axs[1].set_ylim([20, 70])
axs[1].set_ylabel("mem_util")
axs[1].tick_params(axis='both', labelsize=11)

data_memfreq = [(0,3), (5.25,3), (5.25,1), (7.25,1), (7.25,3), (9,3)]
x,y = get_xy(data_memfreq)
axs[2].plot(x,y)
axs[2].set_ylim([0, 5])
axs[2].set_ylabel("mem_freq")
axs[2].tick_params(axis='both', labelsize=11)

data_cpufreq = [(0,3), (9,3)]
x,y = get_xy(data_cpufreq)
axs[3].plot(x,y)
axs[3].set_ylim([0, 5])
axs[3].set_ylabel("cpu_freq")
axs[3].tick_params(axis='both', labelsize=11)

for each_ax in axs:
    each_ax.axvline(x=5, ls='--',lw=0.8, c='grey', alpha=1.0)
    each_ax.axvline(x=5.25, ls='--',lw=0.8, c='grey', alpha=1.0)
    each_ax.axvline(x=5.5, ls='--',lw=0.8, c='grey', alpha=1.0)

    each_ax.axvline(x=7, ls='--',lw=0.8, c='grey', alpha=1.0)
    each_ax.axvline(x=7.25, ls='--',lw=0.8, c='grey', alpha=1.0)
    each_ax.axvline(x=7.5, ls='--',lw=0.8, c='grey', alpha=1.0)

plt.subplots_adjust(top=0.97, left=0.08, right=0.98, bottom=0.03, wspace=0.20, hspace=0.20)

plt.show()