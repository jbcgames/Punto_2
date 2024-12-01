import matplotlib.pyplot as plt
import pandas as pd

# Leer los datos desde un archivo txt
def load_data_from_txt(file_path):
    return pd.read_csv(file_path, delim_whitespace=True)

# Función para generar gráficos para cada kernel
def plot_gpu_time_vs_max_thread(df):
    kernels = [col for col in df.columns if col.startswith("GPU_TIME_KERNEL")]

    for kernel in kernels:
        plt.figure()
        plt.plot(df["MAX_THREAD"], df[kernel], marker='o', label=kernel)
        plt.title(f"Max Thread vs GPU Time ({kernel})")
        plt.xlabel("Max Thread")
        plt.ylabel("GPU Time")
        plt.legend()
        plt.grid()
        plt.show()

# Ruta al archivo txt
file_path = "resultado_p2.txt"  # Cambia esto por la ruta de tu archivo

# Cargar los datos y generar los gráficos
df = load_data_from_txt(file_path)
plot_gpu_time_vs_max_thread(df)
