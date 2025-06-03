def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.actions)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        
        try:
            tau = 0.5
            probs = torch.nn.functional.softmax(q_values / tau, dim=1)  # Q - probs
            action_index = torch.multinomial(probs, num_samples=1).item()
            return self.actions[action_index]
        except IndexError:
            return random.choice(self.actions)  # error - random
