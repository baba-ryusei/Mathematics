import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel1

# パラメータ設定
wavelength = 1.0         # 波長
k = 2 * np.pi / wavelength  # 波数
grid_size = 200
x = np.linspace(-10, 10, grid_size)
y = np.linspace(-10, 10, grid_size)
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)

# 散乱体（半径aの円）
a = 2.0
scatterer_mask = r < a

# 入射波（平面波）
incident_wave = np.exp(1j * k * X)

# 散乱波（Hankel関数を用いた円形散乱の近似解）
scattered_wave = np.zeros_like(X, dtype=complex)
scattered_wave[~scatterer_mask] = hankel1(0, k * r[~scatterer_mask])

# 全波（入射波＋散乱波）
total_wave = incident_wave + scattered_wave

# 可視化
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.title("Real part of total wave")
plt.pcolormesh(X, Y, np.real(total_wave), shading='auto', cmap='RdBu')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Amplitude of total wave")
plt.pcolormesh(X, Y, np.abs(total_wave), shading='auto', cmap='viridis')
plt.colorbar()
plt.tight_layout()
plt.show()
