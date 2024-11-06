# 2024_11_6 范佳林
# 结合单室模型和电缆理论_尝试1
# 将一个神经细胞看作三部分：树突，轴突，胞体
# 树突，轴突使用Cable Theory，胞体采用单室模型
import torch
import math

from matplotlib import pyplot as plt

# ------------参数值------------
# 膜参数
C_m = 1.0 # 膜电容(µF/cm^2)
E_L = -65.0 # 静息电位(mV)

# 电缆通用参数
r_a = 100.0 # (Ω*cm)
g_L_cable = 0.001 # 漏电参数(S/cm^2)

# 树突参数
L_tree = 100.0 # (µm)
dx_tree = 5.0 # (µm)

# 轴突参数
L_axon = 1000.0 # (µm)
dx_axon = 5.0 # (µm)

# 胞体参数
g_L_body = 0.1 # 漏电参数(mS/cm^2)
V_th = -55.0 # 阈值电位
V_reset = -75.0 # 不应期电位
V_spike = 20 # 激发电压

# 衍生参数
A_tree = math.pi * (dx_tree**2) / 4  # 树突的横截面面积(µm^2)
A_axon = math.pi * (dx_axon**2) / 4  # 轴突的横截面面积(µm^2)

# 仿真参数
dx = 20.0 # (µm)
dt = 0.01 # (ms)
t_end = 50.0 # (ms)
t = torch.arange(0, t_end + dt, dt)  # 时间分段
x_tree = torch.arange(0, L_tree + dx, dx)      # 树突上空间分段
Nx_tree = x_tree.size(0)
x_axon = torch.arange(0, L_axon + dx, dx)      # 轴突上空间分段
Nx_axon = x_axon.size(0)
#print(Nx_tree, Nx_axon)

# 树突上输入电流
I_inj_tree = torch.zeros(Nx_tree)
I_inj_tree[0] = 30.0 # 注入电流点
t_inj_tree_start = 0
t_inj_tree_end = 15# (ms)

# 轴突上产生电流
I_inj_axon = torch.zeros(Nx_tree)
I_inj_axon[0] = 0 # 注入电流点
t_inj_axon_start = 0
t_inj_axon_end = 0 # (ms)

# 胞体上电流
I_inj_body = torch.zeros(t.size(0))

# 膜电位初始化
V_tree = E_L * torch.ones(Nx_tree, t.size(0))
V_body = E_L
spike_time = 0
V_axon = E_L * torch.ones(Nx_axon, t.size(0))
count = 0

# ------------仿真循环------------
# 一开始树突收到电流刺激。产生电压差传递到胞体产生电流
for i in range(1, t.size(0) - 1):
    d2Vdx2 = torch.zeros(Nx_tree)
    d2Vdx2[0] = (V_tree[1, i - 1] - 2 * V_tree[0, i - 1] + V_tree[0, i - 1]) / dx**2  # 前向差分
    d2Vdx2[1:Nx_tree - 1] = (V_tree[0:Nx_tree - 2, i - 1] - 2 * V_tree[1:Nx_tree - 1, i - 1] + V_tree[2:Nx_tree, i - 1]) / dx**2
    d2Vdx2[Nx_tree - 1] = (V_tree[Nx_tree - 2, i - 1] - 2 * V_tree[Nx_tree - 1, i - 1] + V_tree[Nx_tree - 1, i - 1]) / dx**2  # 后向差分

    # 根据Cable Theory公式更新膜电位
    if t_inj_tree_start <= t[i] <= t_inj_tree_end:
        V_tree[:, i] = V_tree[:, i - 1] + dt / C_m * (d2Vdx2 / (4 * r_a * A_tree / 1e8) - g_L_cable * (V_tree[:, i - 1] - E_L) + I_inj_tree)
    else:
        V_tree[:, i] = V_tree[:, i - 1] + dt / C_m * (d2Vdx2 / (4 * r_a * A_tree / 1e8) - g_L_cable * (V_tree[:, i - 1] - E_L))

    # 产生电流输入胞体
    I_inj_body[i] = (math.pi * (dx_tree**2) / (4 * r_a)) * ((V_tree[Nx_tree-2, i] - V_tree[Nx_tree-1, i])/dx) * 1e3 # I_long 公式
    dVdt = (-g_L_body * (V_body - E_L) + I_inj_body[i]) / C_m  # Integrate-and-Fire Model公式
    V_body += dVdt * dt  # 膜电位更新

    if V_body >= V_th: # 获得激发
        spike_time =  i * dt # 记录激发时间
        count = i
        V_body = V_reset # 进入不应期
        break

print("胞体获得激发的时间（ms）：",spike_time)
V_axon[0,count] = V_spike # 轴突前段第一个dx获得电压差

#  轴突接受到胞体的激发电压，产生电流
for i in range(count+1, t.size(0) - 1):
    d2Vdx2 = torch.zeros(Nx_axon)
    d2Vdx2[0] = (V_axon[1, i - 1] - 2 * V_axon[0, i - 1] + V_axon[0, i - 1]) / dx ** 2  # 前向差分
    d2Vdx2[1:Nx_axon - 1] = (V_axon[0:Nx_axon - 2, i - 1] - 2 * V_axon[1:Nx_axon - 1, i - 1] + V_axon[2:Nx_axon, i - 1]) / dx ** 2
    d2Vdx2[Nx_axon - 1] = (V_axon[Nx_axon - 2, i - 1] - 2 * V_axon[Nx_axon - 1, i - 1] + V_axon[ Nx_axon - 1, i - 1]) / dx ** 2  # 后向差分
    # 根据Cable Theory公式更新膜电位
    V_axon[:, i] = V_axon[:, i - 1] + dt / C_m * (d2Vdx2 / (4 * r_a * A_axon / 1e8) - g_L_cable * (V_axon[:, i - 1] - E_L))

# ------------绘图，只画了轴突的图像------------
# --- 图1: 距离-膜电位 ---
plt.figure(figsize=(10, 6))
for j in range(0, len(t), 100):
    plt.plot(x_axon, V_axon[:, j].numpy(), label=f"t = {t[j].item():.2f} ms")
plt.xlabel("Distance (µm)")
plt.ylabel("Membrane Potential (mV)")
plt.title("Cable Equation Simulation (Distance as x-axis)")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), prop={'size': 5})  # 图例在图外
plt.show()

# --- 图2: 时间-膜电位 ---
plt.figure(figsize=(10, 6))
t = t[:-1]
for i in range(5, Nx_axon, 5):  # 每隔5个空间点绘制一次
    plt.plot(t.numpy(), V_axon[i, :-1].numpy(), label=f"x = {x_axon[i].item()} µm")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.title("Cable Equation Simulation (Time as x-axis)")
plt.legend()
plt.show()