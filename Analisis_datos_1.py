import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Leer datos desde un archivo de texto
def cargar_datos(archivo):
    # Leer el archivo tratando tanto tabulaciones como múltiples espacios
    return pd.read_csv(archivo, delim_whitespace=True)

# Cargar datos
archivo = "Punto1.txt"
df = cargar_datos(archivo)

# Validar columnas
print("Encabezados del archivo:", df.columns)
if "WORK_SIZE" not in df.columns:
    raise ValueError("La columna 'WORK_SIZE' no está presente en el archivo. Verifica los encabezados.")

# Filtrar datos por kernel ejecutado
kernels = ["GPU_TIME_KERNEL_V1", "GPU_TIME_KERNEL_V2", "GPU_TIME_KERNEL_V3"]
gpu_kernel_data = []

for kernel in kernels:
    filtered = df[df[kernel] != 0].copy()
    filtered["GPU_KERNEL"] = kernel
    gpu_kernel_data.append(filtered)

gpu_kernel_df = pd.concat(gpu_kernel_data)

# Agrupar por WORK_SIZE y GPU_KERNEL y calcular promedios
grouped = gpu_kernel_df.groupby(["WORK_SIZE", "GPU_KERNEL"]).mean().reset_index()

# Crear gráficos comparativos de CPU_TIME vs GPU_TIME_TOTAL por kernel
for kernel in kernels:
    plt.figure(figsize=(10, 6))
    kernel_data = grouped[grouped["GPU_KERNEL"] == kernel]
    plt.plot(kernel_data["WORK_SIZE"], kernel_data["CPU_TIME"], marker='o', label='CPU_TIME')
    plt.plot(kernel_data["WORK_SIZE"], kernel_data["GPU_TIME_TOTAL"], marker='x', linestyle='--', label='GPU_TIME_TOTAL')

    plt.xlabel('WORK_SIZE ')
    plt.ylabel('Time (seconds)')
    plt.title(f'CPU Time vs GPU Total Time ({kernel})')
    plt.legend()
    plt.grid(True)
    plt.show()
