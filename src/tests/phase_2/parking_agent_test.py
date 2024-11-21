import argparse
import logging
import random
import sys
import warnings
import numpy as np
from pathlib import Path
from typing import Final, Any

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

import gymnasium as gym
from smarts.env.gymnasium.wrappers.parking_agent import ParkingAgent

SMARTS_REPO_PATH = Path(__file__).parents[1].absolute()
sys.path.insert(0, str(SMARTS_REPO_PATH))
from examples.tools.argument_parser import minimal_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, ActionSpaceType, AgentType
from smarts.core.utils.episodes import episodes
from smarts.sstudio.scenario_construction import build_scenarios
from smarts.core.scenario import Scenario

AGENT_ID: Final[str] = "Agent"


class LearningAgent:
    """Agente de Q-learning optimizado para aparcamiento, que utiliza LiDAR con velocidad angular fija."""

    def __init__(self, epsilon=0.2, alpha=0.1, gamma=0.99):
        self.epsilon = epsilon  # Probabilidad de exploración en epsilon-greedy
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento
        self.q_table = {}  # Tabla Q para almacenar los valores de estado-acción
        self.actions = [-2, -1, 0, 1, 2]  # Aceleraciones lineales posibles

    def discretize(self, value, step=0.25, max_value=10.0):
        """Discretiza un valor continuo al múltiplo más cercano de 'step'.

        Args:
            value (float): Valor continuo a discretizar.
            step (float): Tamaño del intervalo de discretización.
            max_value (float): Límite máximo (los valores mayores se limitan).

        Returns:
            float: Valor discretizado al múltiplo más cercano de 'step'.
        """
        # Limitar el valor a [-max_value, max_value]
        value = min(max(value, -max_value), max_value)
        # Redondear al múltiplo más cercano de step
        return round(value / step) * step

    def get_state(self, observation):
        """Extrae y discretiza el estado basado en el LiDAR y la posición del vehículo."""
        lidar_data = observation["lidar_point_cloud"]["point_cloud"]
        car_pose = np.array(observation["ego_vehicle_state"]["position"])

        # Asignar inf a obstáculos ausentes.
        lidar_data[np.all(lidar_data == [0, 0, 0], axis=1)] = float('inf')
        relative_points = lidar_data - car_pose
        distances = np.linalg.norm(relative_points, axis=1)

        lidar_length = len(distances)
        index_90 = lidar_length // 4
        index_270 = (3 * lidar_length) // 4

        distance_90 = self.discretize(distances[index_90])
        distance_270 = self.discretize(distances[index_270])

        distance_difference = self.discretize(distance_270 - distance_90)

        return float(distance_difference)

    def choose_action(self, state):
        """Selecciona una acción basada en la política epsilon-greedy."""
        if np.random.rand() < self.epsilon:
            # Explorar: Elegir una acción aleatoria
            return np.random.choice(self.actions)

        # Explotar: Elegir la mejor acción conocida
        # print(f"Diferencia de distancias: {state}")
        if state not in self.q_table:
            # Inicializar valores Q para acciones en el estado si no existen
            self.q_table[state] = {action: 0.0 for action in self.actions}

        # Devolver la acción con el valor Q más alto en este estado
        return max(self.q_table[state], key=self.q_table[state].get)

    def act(self, observation):
        """Genera una acción basada en la observación del entorno."""
        state = self.get_state(observation)
        action = self.choose_action(state)
        # print(f"Accion elegida: {action}       En estado: {state}")
        return np.array([action, 0.0])

    def learn(self, state, action, reward, next_state):
        """Actualiza la tabla Q según la fórmula de Q-learning."""
        # Inicializar los estados en la tabla Q si no están presentes
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0.0 for action in self.actions}

        # Calcular el valor Q futuro máximo
        max_future_q = max(self.q_table[next_state].values())

        # print(f"Recompensa obtenida: {reward}")

        # Actualizar la tabla Q
        current_q = self.q_table[state].get(action, 0.0)
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)



def main(scenarios, headless, num_episodes=200, max_episode_steps=None):
    agent_interface = AgentInterface(
        action=ActionSpaceType.Direct,
        # max_episode_steps=max_episode_steps,
        max_episode_steps=200,
        neighborhood_vehicle_states=True,
        waypoint_paths=True,
        road_waypoints=True,
        drivable_area_grid_map=True,
        occupancy_grid_map=True,
        top_down_rgb=True,
        lidar_point_cloud=True,
        accelerometer=True,
        lane_positions=True,
        signals=True,
    )

    # target_position = (23.25,100.0,0.0)

    env = gym.make(
        "smarts.env:hiway-v1",
        scenarios=scenarios,
        agent_interfaces={AGENT_ID: agent_interface},
        headless=headless,
    )

    env = ParkingAgent(env)
    

    for episode in episodes(n=num_episodes):
        agent = LearningAgent()
        observation, _ = env.reset()
        episode.record_scenario(env.unwrapped.scenario_log)

        terminated = False
        while not terminated:
            state = agent.get_state(observation)
            action = agent.act(observation)

            next_observation, reward, terminated, truncated, info = env.step((action[0],action[1]))
            next_state = agent.get_state(next_observation)

            agent.learn(state, action[0], reward, next_state)

            observation = next_observation
            episode.record_step(observation, reward, terminated, truncated, info)

    env.close()


if __name__ == "__main__":
    parser = minimal_argument_parser(Path(__file__).stem)
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(SMARTS_REPO_PATH / "scenarios" / "sumo" / "loop"),
            str(SMARTS_REPO_PATH / "scenarios" / "sumo" / "figure_eight"),
        ]

    build_scenarios(scenarios=args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=100,
        max_episode_steps=200,
    )
