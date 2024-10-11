import numpy as np
import matplotlib.pyplot as plt

# Fungsi f(x) hubungan antara AR dan CL
f = lambda x: (6.28 * x**2) / (x + 2)

# Turunan f'(x)
df = lambda x: (6.28 * x**2 + 25.12 * x) / (x + 2)**2

# Tebakan awal
x0 = 5

# Fungsi untuk metode Newton-Raphson
def newton_raphson(f, df, x0, tolerance=0.001, max_iterations=100):
    x = x0
    x_vals = [x]  # Menyimpan nilai x pada setiap iterasi untuk visualisasi
    for i in range(max_iterations):
        fx = f(x)
        dfx = df(x)
        if abs(fx) < tolerance:
            return x, i, x_vals  # Mengembalikan akar, jumlah iterasi, dan semua nilai x
        x = x - fx / dfx
        x_vals.append(x)  # Simpan nilai x baru
    return x, max_iterations, x_vals  # Mengembalikan hasil setelah iterasi maksimum

# Menjalankan metode Newton-Raphson
akar, iterasi, x_vals = newton_raphson(f, df, x0)

# Menampilkan hasil
print("Hasil akar persamaan:", akar)
print("Jumlah iterasi:", iterasi)

# Visualisasi menggunakan matplotlib
# Rentang nilai x untuk grafik
x_range = np.linspace(0.1, 10, 400) 
y_range = f(x_range)

# Plot fungsi f(x)
plt.plot(x_range, y_range, label="Fungsi Persamaan AR")

# Plot iterasi Newton-Raphson
x_iter = np.array(x_vals)
y_iter = f(x_iter)
plt.plot(x_iter, y_iter, 'ro-', label="Iterasi menggunakan Newton-Raphson")

# Tambahkan informasi visual lainnya
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(akar, color='green', linestyle='--', label=f"Hasil akar persamaan: {akar:.4f}")
plt.title("Newton-Raphson Method")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)

# Tampilkan plot
plt.show()
