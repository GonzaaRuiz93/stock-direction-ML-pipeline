

import matplotlib.pyplot as plt
import numpy as np


def Get_comparative_figure(data1, data2, label1="Modelo 1", label2="Modelo 2"):
    # 1. Filtramos nombres (sin 'trades')
    names = [k for k in data1.keys() if 'trades' not in k]
    
    # 2. Convertimos np.float64 a float nativo (esto evita conflictos con tkinter)
    val1 = [float(data1[k]) for k in names]
    val2 = [float(data2[k]) for k in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    # Usamos el estilo 'seaborn' para que se vea más moderno en VS Code
    plt.style.use('seaborn-v0_8-muted') 
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rects1 = ax.bar(x - width/2, val1, width, label=label1, color="skyblue", edgecolor='white')
    rects2 = ax.bar(x + width/2, val2, width, label=label2, color="salmon", edgecolor='white')
    
    # Configuración de ejes
    ax.set_title("Comparativa de Modelos", fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()

    # Función para etiquetas
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()
    # Cerramos explícitamente para limpiar el hilo
    plt.close(fig)
