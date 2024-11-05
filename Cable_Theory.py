# 2024_11_5 范佳林
# 改写老师提供的Julia代码
import matplotlib.pyplot as plt
import torch
import math

# ------------参数值------------
# 膜参数
C_m = 1.0 # (µF/cm^2)
g_L = 0.001 # (S/cm^2)
E_L = -65.0 # (mV)

# 电缆参数
d = 5.0 # (µm)
L = 1000.0 # (µm)
r_a = 100.0 # (Ω*cm)

# 衍生参数
A = math.pi * (d**2) / 4  # 电缆的横截面面积(µm^2)
B = math.sqrt(1 / (4 * r_a * g_L * A / 1e8))  # 空间常数(µm)
print('空间参数为(µm)：', B)

# 仿真参数
dx = 20.0 # (µm)
dt = 0.01 # (ms)
t_end = 50.0 # (ms)
t = torch.arange(0, t_end + dt, dt)  # 时间分段
x = torch.arange(0, L + dx, dx)      # 空间分段
Nx = x.size(0)

# 输入电流
I_inj = torch.zeros(Nx)
I_inj[int(round(Nx / 2))] = 20.0  # 当前电流注入点
t_inj_start = 0
t_inj_end = 10 # (ms)

# 膜电位初始化
V = E_L * torch.ones(Nx, t.size(0))

# 仿真循环
for i in range(1, t.size(0) - 1):
    d2Vdx2 = torch.zeros(Nx)
    d2Vdx2[0] = (V[1, i - 1] - 2 * V[0, i - 1] + V[0, i - 1]) / dx**2  # 前向差分
    d2Vdx2[1:Nx - 1] = (V[0:Nx - 2, i - 1] - 2 * V[1:Nx - 1, i - 1] + V[2:Nx, i - 1]) / dx**2
    d2Vdx2[Nx - 1] = (V[Nx - 2, i - 1] - 2 * V[Nx - 1, i - 1] + V[Nx - 1, i - 1]) / dx**2  # 后向差分

    # 根据Cable Theory公式更新膜电位
    if t_inj_start <= t[i] <= t_inj_end:
        V[:, i] = V[:, i - 1] + dt / C_m * (d2Vdx2 / (4 * r_a * A / 1e8) - g_L * (V[:, i - 1] - E_L) + I_inj)
    else:
        V[:, i] = V[:, i - 1] + dt / C_m * (d2Vdx2 / (4 * r_a * A / 1e8) - g_L * (V[:, i - 1] - E_L))

# --- 图1: 距离-膜电位 ---
plt.figure(figsize=(10, 6))
for j in range(0, len(t), 150):
    plt.plot(x, V[:, j].numpy(), label=f"t = {t[j].item():.2f} ms")
plt.xlabel("Distance (µm)")
plt.ylabel("Membrane Potential (mV)")
plt.title("Cable Equation Simulation (Distance as x-axis)")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), prop={'size': 6})  # 图例在图外
plt.show()

# --- 图2: 时间-膜电位 ---
plt.figure(figsize=(10, 6))
t = t[:-1]
for i in range(0, Nx // 2, 5):  # 每隔5个空间点绘制一次
    plt.plot(t.numpy(), V[i, :-1].numpy(), label=f"x = {x[i].item()} µm")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.title("Cable Equation Simulation (Time as x-axis)")
plt.legend()
plt.show()