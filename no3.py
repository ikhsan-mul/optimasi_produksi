import numpy as np
import matplotlib.pyplot as plt

# Inisialisasi parameter awal
x1 = 8
x2 = 4
alpha = 0.1  # Learning rate
tolerance = 0.001  # Toleransi kesalahan
max_iter = 1000  # Batas maksimal iterasi

# Simpan riwayat nilai x1, x2, dan performa P untuk visualisasi
history_x1 = []
history_x2 = []
history_P = []

# Fungsi performa P(x1, x2)
def P(x1, x2):
    return -(x1 - 10)**2 - (x2 - 5)**2 + 50

# Fungsi gradien dari P(x1, x2)
def gradient(x1, x2):
    dP_dx1 = -2 * (x1 - 10)
    dP_dx2 = -2 * (x2 - 5)
    return dP_dx1, dP_dx2

# Gradient Descent
for i in range(max_iter):
    # Hitung gradien saat ini
    dP_dx1, dP_dx2 = gradient(x1, x2)
    
    # Simpan nilai x1 dan x2 sebelumnya
    prev_x1, prev_x2 = x1, x2
    
    # Update nilai x1 dan x2
    x1 = x1 - alpha * dP_dx1
    x2 = x2 - alpha * dP_dx2
    
    # Simpan riwayat untuk visualisasi
    history_x1.append(x1)
    history_x2.append(x2)
    history_P.append(P(x1, x2))
    
    # Hitung perubahan pada x1 dan x2
    delta_x1 = abs(x1 - prev_x1)
    delta_x2 = abs(x2 - prev_x2)
    
    # Jika perubahan lebih kecil dari toleransi, berhenti
    if delta_x1 < tolerance and delta_x2 < tolerance:
        break

# Cetak hasil
print(f"Nilai optimal x1: {x1}")
print(f"Nilai optimal x2: {x2}")
print(f"Jumlah iterasi: {i}")

# Visualisasi hasil menggunakan matplotlib
plt.figure(figsize=(12, 6))

# Plot x1 dan x2 selama iterasi
plt.subplot(1, 2, 1)
plt.plot(history_x1, label='x1 (Panjang Sayap)')
plt.plot(history_x2, label='x2 (Sudut Serang)')
plt.xlabel('Iterasi')
plt.ylabel('Nilai')
plt.title('Perubahan x1 dan x2 selama iterasi')
plt.legend()

# Plot performa P selama iterasi
plt.subplot(1, 2, 2)
plt.plot(history_P, color='g', label='Performa P(x1, x2)')
plt.xlabel('Iterasi')
plt.ylabel('Performa')
plt.title('Performa P(x1, x2) selama iterasi')
plt.legend()

# Tampilkan plot
plt.tight_layout()
plt.show()
