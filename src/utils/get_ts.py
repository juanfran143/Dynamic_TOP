import numpy as np
import matplotlib.pyplot as plt


def logistic_function(x, betas):
    return 1 / (1 + np.exp(-np.dot(betas, x)))


LOW = [
    {0: 0, 1: 0.0, 2: 0.0, 3: 0, 4: 0},
    {0: 0, 1: -0.1, 2: -0.2, 3: -0.3, 4: -0.4},
    {0: -1, 1: -0.8, 2: -0.6, 3: -0.4, 4: -0.2},
    {0: 1, 1: 1.1, 2: 1.2, 3: 1.3, 4: 1.4}
]

MEDIUM = [
    {0: 0, 1: 0.0, 2: 0.0, 3: 0, 4: 0},
    {0: 0, 1: -0.2, 2: -0.4, 3: -0.6, 4: -0.8},
    {0: -1.2, 1: -1, 2: -0.8, 3: -0.6, 4: -0.4},
    {0: 1.2, 1: 1.4, 2: 1.6, 3: 1.8, 4: 2}
]

HIGH = [
    {0: 0, 1: 0.0, 2: 0.0, 3: 0, 4: 0},
    {0: 0, 1: -0.5, 2: -1, 3: -1.5, 4: -2},
    {0: -2, 1: -1.5, 2: -1, 3: -0.8, 4: -0.5},
    {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
]


# Convertir los diccionarios en listas para facilitar el cálculo
betas = [list(beta.values()) if isinstance(beta, dict) else beta for beta in MEDIUM]

# Nuevo rango para variar beta 1
beta_3_range = np.linspace(-1, 1, 100)  # Variando de -1 a 1

# Nuevos ajustes en la interpretación de los datos de betas
beta_0 = list(MEDIUM[0].values())  # Beta 0 (intercept)
beta_1 = list(MEDIUM[1].values())  # Beta 1
beta_2 = list(MEDIUM[2].values())  # Beta 2
beta_3 = list(MEDIUM[3].values())  # Beta 3

# Preparación para la generación de gráficos
fig, axs = plt.subplots(1, 5, figsize=(20, 4))  # 5 gráficos para 5 tipos de nodos

# Generación de gráficos para cada tipo de nodo
for i in range(5):
    for beta_1_val in [-1, 1]:
        for beta_2_val in [-1, 1]:
            # Valores de la función logística para la combinación actual de beta 2 y beta 3
            y_values = [logistic_function([1, beta_1_val, beta_2_val, val],
                                          [beta_0[i], beta_1[i], beta_2[i], beta_3[i]])
                        for val in beta_3_range]
            axs[i].plot(beta_3_range, y_values, label=f'Beta 1 = {beta_1_val}, Beta 2 = {beta_2_val}')

    axs[i].set_title(f'Type {i+1}')
    axs[i].set_xlabel('Beta 3 Value')
    axs[i].set_ylabel('Function Output')
    axs[i].legend()

plt.tight_layout()
plt.show()


