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

    def __init__(self, epsilon=0.99, min_epsilon=0.01, decay_rate=0.97, alpha=0.12, gamma=0.9):
        self.epsilon = epsilon  # Probabilidad inicial de exploración
        self.min_epsilon = min_epsilon  # Valor mínimo de epsilon
        self.decay_rate = decay_rate  # Tasa de decremento para epsilon
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento
        self.q_table = {} # Tabla Q para almacenar los valores de estado-acción
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
        """Extrae y discretiza el estado basado en el LiDAR y la velocidad del vehículo."""
        lidar_data = observation["lidar_point_cloud"]["point_cloud"]
        # car_pose = np.array(observation["ego_vehicle_state"]["position"])
        # heading = observation["ego_vehicle_state"]["heading"]

        # heading_deg = np.degrees(heading)

        # Asignar inf a obstáculos ausentes.
        # lidar_data[np.all(lidar_data == [0, 0, 0], axis=1)] = float('inf')
        # relative_points = lidar_data - car_pose
        # distances = np.linalg.norm(relative_points, axis=1)

        # lidar_resolution = 360 / len(distances)
        # index_90 = int(round(heading_deg / lidar_resolution))
        # index_270 = int(round((heading_deg + 180) / lidar_resolution))

        # distance_90 = self.discretize(distances[index_90])
        # distance_270 = self.discretize(distances[index_270])

        distances = np.linalg.norm(lidar_data, axis=1)
        lidar_resolution = 360 / len(distances)
        index_90 = int(round(90 / lidar_resolution))
        index_270 = int(round(270 / lidar_resolution))
        distance_90 = self.discretize(distances[index_90])
        # print(f"Distancia delante: {distance_90}")
        distance_270 = self.discretize(distances[index_270])
        # print(f"Distancia detras: {distance_270}")
        distance_difference = self.discretize(distance_270 - distance_90)

        velocity = observation['ego_vehicle_state']['speed']
        discretized_velocity = self.discretize(velocity, step=1, max_value=20)

        return (distance_difference, discretized_velocity)

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

        # Actualizar la tabla Q
        current_q = self.q_table[state].get(action, 0.0)
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

    def decay_epsilon(self):
        """Reduce epsilon según la tasa de decremento."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
    
    def print_q_table(self):
        """Imprime la tabla Q completa."""
        for state, actions in self.q_table.items():
            print(f"State: {state}")
            for action, value in actions.items():
                print(f"  Action: {action}, Q-value: {value}")



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
    agent = LearningAgent()

    for episode in episodes(n=num_episodes):
        observation, _ = env.reset()
        episode.record_scenario(env.unwrapped.scenario_log)

        terminated = False
        # t_reward = 0
        while not terminated:
            state = agent.get_state(observation)
            action = agent.act(observation)

            next_observation, reward, terminated, truncated, info = env.step((action[0],action[1]))
            next_state = agent.get_state(next_observation)

            agent.learn(state, action[0], reward, next_state)

            observation = next_observation
            # t_reward = t_reward + reward
            episode.record_step(observation, reward, terminated, truncated, info)
        # print(f"Recompensa de episodio: {t_reward}")
        agent.decay_epsilon()
    agent.print_q_table()

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
        num_episodes=250,
        max_episode_steps=200,
    )
