import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel1

# 基本設定
grid_size = 100
x = np.linspace(-10, 10, grid_size)
y = np.linspace(-10, 10, grid_size)
X, Y = np.meshgrid(x, y)
r_grid = np.sqrt(X**2 + Y**2)
wavelength = 1.0
k = 2 * np.pi / wavelength

# 散乱体（中心原点、半径a）
a = 2.0
scatterer_mask = r_grid < a

# 観測点の配置（円周上）
num_sensors = 60
sensor_radius = 8.0
sensor_angles = np.linspace(0, 2*np.pi, num_sensors, endpoint=False)
sensor_x = sensor_radius * np.cos(sensor_angles)
sensor_y = sensor_radius * np.sin(sensor_angles)

# 散乱波の生成（正問題の簡易近似）
def generate_scattered_wave(x0, y0):
    r = np.sqrt((X - x0)**2 + (Y - y0)**2)
    wave = hankel1(0, k * r)
    wave[r < 1e-3] = 0  # avoid singularity
    return wave

# 観測波データ（散乱体から各センサへの散乱波）
sensor_data = []
for x_s, y_s in zip(sensor_x, sensor_y):
    r = np.sqrt((X - x_s)**2 + (Y - y_s)**2)
    wave = hankel1(0, k * r)
    wave[r < 1e-3] = 0
    # 散乱体がある位置の散乱波の強さ（仮に中心を使う）
    sensor_data.append(np.real(wave[grid_size//2, grid_size//2]))
sensor_data = np.array(sensor_data)

# 🔄 逆問題：後方投影的再構成
reconstruction = np.zeros_like(X, dtype=float)
for i, (x_s, y_s) in enumerate(zip(sensor_x, sensor_y)):
    r = np.sqrt((X - x_s)**2 + (Y - y_s)**2)
    wave = hankel1(0, k * r)
    wave[r < 1e-3] = 0
    reconstruction += np.real(wave) * sensor_data[i]

# 正規化
reconstruction = (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min())

# 可視化
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Ground Truth (Scatterer)")
plt.pcolormesh(X, Y, scatterer_mask, shading='auto', cmap='Greys')
plt.scatter(sensor_x, sensor_y, color='red', s=10, label='Sensors')
plt.axis('equal')
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Reconstruction (Backprojection)")
plt.pcolormesh(X, Y, reconstruction, shading='auto', cmap='inferno')
plt.axis('equal')

plt.tight_layout()
plt.show()
