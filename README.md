# Laboratorio de Análisis de Rendimiento GPU

Este repositorio contiene el código y las instrucciones necesarias para realizar un análisis de rendimiento en GPU comparando tiempos de ejecución en CPU y GPU para diferentes configuraciones.

## Objetivo

1. **Análisis de tamaño del vector:**
   - Variar el tamaño del vector (`WORK_SIZE`) entre $2^{10}$ (1024) y $2^{30}$ (1073741824).
   - Comparar el tiempo de ejecución en CPU (`CPU_TIME`) con el tiempo total de GPU (`GPU_TIME_TOTAL`).
   - Analizar cómo cambian los tiempos reportados por la GPU:
     - Tiempo total de GPU (`GPU_TIME_TOTAL`).
     - Tiempo del kernel (`GPU_TIME_KERNEL`).

2. **Análisis de número de hilos por bloque:**
   - Fijar el tamaño del vector en $2^{28}$.
   - Variar el número de hilos por bloque (`MAX_THREAD`) desde 32 hasta 1024 en pasos de 4.
   - Analizar el impacto de este parámetro en el tiempo de ejecución del kernel (`GPU_TIME_KERNEL`).

## Pasos para la ejecución

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/jbcgames/Punto_2.git
   ```

2. Acceder al directorio del proyecto:
   ```bash
   cd Punto_2
   ```

3. Compilar los archivos CUDA:
   ```bash
   nvcc kernel_editado.cu -o kernel
   nvcc kernel_editado_p2.cu -o kernel_p2
   ```

4. Ejecutar los binarios generados:
   - Para el primer análisis:
     ```bash
     ./kernel > Punto1.txt
     ```
   - Para el segundo análisis:
     ```bash
     ./kernel_p2 > resultado_p2.txt
     ```

5. Analizar los resultados utilizando los scripts de Python:
   - Para el primer análisis:
     ```bash
     python Analisis_datos_1.py
     ```
   - Para el segundo análisis:
     ```bash
     python Analisis_datos_2.py
     ```

## Resultados esperados

- Un archivo `Punto1.txt` que contiene los tiempos de ejecución para diferentes tamaños de vector.
- Un archivo `resultado_p2.txt` con los tiempos obtenidos al variar el número de hilos por bloque.
- Gráficas generadas por los scripts de Python que facilitan el análisis comparativo de los datos.

## Notas
- Asegúrese de tener instalado el compilador CUDA (`nvcc`) y las dependencias de Python necesarias para ejecutar los scripts de análisis.
- Consulte los códigos fuente `kernel_editado.cu` y `kernel_editado_p2.cu` para más detalles sobre la instrumentación del código y la recolección de datos.

---
**Autor:** Miguel Angel Alvarez Guzman

