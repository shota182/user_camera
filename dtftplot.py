import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv("/home/sskr3/ros_ws/user_camera/output.csv")
# data = pd.read_csv("/home/sskr3/ros_ws/user_camera/dtft.csv")

original = data.T[0:3]
dtft = data.T[3:]

x = np.arange(0, 1, 0.01)
X = np.arange(0, 2*np.pi, 2*np.pi/100)
f = np.arange(100)

# print(original.iloc[0])
original_x = np.array([np.real(complex(i)) for i in original.iloc[0]]) #
original_y = np.array([np.real(complex(i)) for i in original.iloc[1]])
# original_z = np.array([np.real(complex(i))+9.81 for i in original.iloc[2]])
original_z = np.array([np.real(complex(i)) for i in original.iloc[2]])
fft = np.fft.fft(original_z)

dtft_x = np.array([np.abs(complex(i)) for i in dtft.iloc[0]]) #
dtft_y = np.array([np.abs(complex(i)) for i in dtft.iloc[1]])
dtft_z = np.array([np.abs(complex(i)) for i in dtft.iloc[2]])

plt.rcParams["font.size"] = 15
plt.figure()
plt.grid()
# plt.ylim(-15,15)
plt.ylim(0,120)
plt.xlabel("time[sec]", fontsize=30)
# plt.ylabel("acceleration[m/s^2]", fontsize=30)
plt.ylabel("|X(ω)|", fontsize=30)
# plt.plot(x, original_x, color="red", label="x")
# plt.plot(x, original_y, color="blue", label="y")
# plt.plot(x, original_z, color="green", label="z")
# plt.plot(X[:51], dtft_x[:51], color="red")
# plt.plot(X[:51], dtft_y[:51], color="blue")
# plt.plot(X[:51], dtft_z[:51], color="green")
plt.plot(f[:51], dtft_x[:51], color="red", label="x")
plt.plot(f[:51], dtft_y[:51], color="blue", label="y")
plt.plot(f[:51], dtft_z[:51], color="green", label="z")
# plt.plot(f[:51], fft[:51], color="green")
plt.legend()
plt.show()