
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def Get_comparative_figure(data1, data2, label1="Model 1", label2="Model 2", filename="Comparativa_de_Modelos.png"):
    # 1. Filtramos nombres (sin 'trades')
    names = [k for k in data1.keys() if 'trades' not in k]
    
    # 2. Convertimos np.float64 a float nativo
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
    ax.set_title("Model Comparison", fontsize=14, fontweight='bold', pad=20)
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

    current_dir = Path(__file__).resolve().parent
    save_path = current_dir / filename

    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    print(f"Graph generated in: {filename}")

    plt.show()

    # Cerramos explícitamente para limpiar el hilo
    plt.close(fig)
