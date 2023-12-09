import numpy as np
import matplotlib.pyplot as plt


def logistic_function(x, betas):
    return 1 / (1 + np.exp(-np.dot(betas, x)))


HIGHTEST = [
    {0: 0, 1: 0.0, 2: 0.0, 3: 0, 4: 0},
    [-8.9, -7.3, -6.9, -0.2, -0.05],
    {0: -8.2, 1: -7.8, 2: -5.5, 3: -0.6, 4: -0.3},
    {0: -8.5, 1: -7.5, 2: -5.2, 3: -0.5, 4: -0.2}
]

# Convertir los diccionarios en listas para facilitar el cálculo
betas = [list(beta.values()) if isinstance(beta, dict) else beta for beta in HIGHTEST]

# Nuevo rango para variar beta 1
beta_1_range = np.linspace(-1, 1, 100)  # Variando de -1 a 1

# Nuevos ajustes en la interpretación de los datos de betas
beta_0 = list(HIGHTEST[0].values())  # Beta 0 (intercept)
beta_1 = HIGHTEST[1]  # Beta 1
beta_2 = list(HIGHTEST[2].values())  # Beta 2
beta_3 = list(HIGHTEST[3].values())  # Beta 3

# Preparación para la generación de gráficos
fig, axs = plt.subplots(1, 5, figsize=(20, 4))  # 5 gráficos para 5 tipos de nodos

# Generación de gráficos para cada tipo de nodo
for i in range(5):
    for beta_2_val in [0, 1]:
        for beta_3_val in [0, 1]:
            # Valores de la función logística para la combinación actual de beta 2 y beta 3
            y_values = [logistic_function([1, val, beta_2_val, beta_3_val],
                                          [beta_0[i], beta_1[i], beta_2[i], beta_3[i]])
                        for val in beta_1_range]
            axs[i].plot(beta_1_range, y_values, label=f'Beta 2 = {beta_2_val}, Beta 3 = {beta_3_val}')

    axs[i].set_title(f'Type {i+1}')
    axs[i].set_xlabel('Beta 1 Value')
    axs[i].set_ylabel('Function Output')
    axs[i].legend()

plt.tight_layout()
plt.show()


