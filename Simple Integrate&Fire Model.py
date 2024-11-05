# 2024_11_5 范佳林
# 改写老师提供的Julia代码
import torch
import matplotlib.pyplot as plt
# ------------参数值------------
C_m = 1.0   # Membrane capacitance (µF/cm^2)
g_L = 0.1   # Leak conductance (mS/cm^2)
E_L = -70.0 # Leak reversal potential (mV)
V_th = -55.0 # Threshold potential (mV)
V_reset = -75.0 # Reset potential (mV)
V_spike = 20 # Spike peak
tau_ref = 2.0 # Refractory period (ms)

# ------------仿真参数------------
dt = 0.1      # Time step (ms)
t_end = 100.0 # Total simulation time (ms)
t = torch.arange(0, t_end + dt, dt)

# ------------恒定输入电流------------
I_inj = 2 # Constant injected current (µA/cm^2)

# ------------初始值------------
V = E_L # 初始膜电位
t_last_spike = -tau_ref # 上次激发的时间（初始化到仿真开始以前）

# 开始仿真
spike_times = torch.empty(0, dtype=torch.float64) # 存储激发的时刻
V_trace = torch.empty(0, dtype=torch.float64) # 为绘图存储膜电位
for i in range(0, t.size(0)):
    if t[i] > t_last_spike + tau_ref: # 在不应期内膜电位不会有变化
        dVdt = (-g_L * (V - E_L) + I_inj) / C_m # Integrate-and-Fire Model公式
        V += dVdt * dt # 膜电位更新

    if V >= V_th: # 检查膜电位是否到了激发的阈值
        spike_times = torch.cat((spike_times, t[i].unsqueeze(0))) # 记录下激发的时刻
        V = V_reset # 膜电位恢复不应期内的值
        t_last_spike = t[i]

    V_trace = torch.cat((V_trace, torch.tensor(V).unsqueeze(0))) # 记录膜电位

# ------------绘图部分------------
plt.figure(figsize=(10, 6))

# 绘制膜电位曲线
plt.plot(t.numpy(), V_trace.numpy(), label="Membrane Potential", color="blue")

# 绘制激发事件
plt.scatter(spike_times.numpy(), [V_th] * spike_times.size(0), color="red", marker='x', s=50, label="Spikes")

# 设置图表标题和坐标轴标签
plt.title("Integrate-and-Fire Neuron Simulation (Matplotlib)")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend(loc="lower left")
plt.grid()

# 显示图表
plt.show()