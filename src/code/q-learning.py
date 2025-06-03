# Hiperparametros clave 
alpha = 0.2  # Tasa de aprendizaje 
gamma = 0.9  # Factor de descuento 
epsilon = 0.99  # Probabilidad inicial de exploracion 
min_epsilon = 0.0  # Limite inferior para exploracion 
decay_rate = 0.99  # Decaimiento de epsilon por episodio 
 
# Ecuacion de actualizacion Q-Learning 
q_table[state][action] += alpha * (reward + gamma * max(q_table[next_state]) - q_table[state][action]) 
