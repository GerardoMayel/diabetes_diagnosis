import numpy as np


def gradient_descent(X, y, theta, alpha, num_iterations):
    """
    Realiza el descenso del gradiente para minimizar la función de costo J(theta).

    Argumentos:
    X -- matriz de características de entrada, de forma (n_features, m_examples)
    y -- vector de salida esperado, de forma (1, m_examples)
    theta -- parámetros del modelo, de forma (n_features, 1)
    alpha -- tasa de aprendizaje
    num_iterations -- número de iteraciones para realizar el descenso del gradiente

    Devuelve:
    theta -- parámetros del modelo optimizados
    cost_history -- historial de los valores de la función de costo J(theta) a medida que se realiza el descenso del gradiente
    """

    m = y.shape[1]  # número de ejemplos de entrenamiento
    cost_history = []

    for i in range(num_iterations):
        # Calcula la derivada parcial de la función de costo con respecto a cada parámetro theta
        z = np.dot(theta.T, X)
        h = 1 / (1 + np.exp(-z))  # función sigmoide
        dtheta = np.dot(X, (h - y).T) / m

        # Actualiza los parámetros del modelo theta
        theta -= alpha * dtheta

        # Calcula el valor actual de la función de costo y almacena su historial
        cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        cost_history.append(cost)

    return theta, cost_history
