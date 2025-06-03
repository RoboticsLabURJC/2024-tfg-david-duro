def choose_action(self, state):
        """Selecciona una accion basada en la politica epsilon-greedy."""
        if np.random.rand() < self.epsilon:
            # Explorar: Elegir una accion aleatoria
            return np.random.choice(self.actions)

        # Explotar: Elegir la mejor accion conocida
        if state not in self.q_table:
            # Inicializar valores Q para acciones en el estado si no existen
            self.q_table[state] = {action: 0.0 for action in self.actions}

        # Devolver la accion con el valor Q mas alto en este estado
        return max(self.q_table[state], key=self.q_table[state].get)
