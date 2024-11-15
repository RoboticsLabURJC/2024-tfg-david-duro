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


class LearningAgent(Agent):
    """Agente de Q-learning mejorado para aparcamiento, que utiliza LiDAR y distancia al objetivo."""

    def __init__(self, target_position, epsilon=0.1, alpha=0.1, gamma=0.99):
        self.target_position = np.array(target_position)
        self.epsilon = epsilon  # Probabilidad de exploración en epsilon-greedy
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento
        self.q_table = {}  # Tabla Q para almacenar los valores de estado-acción

    def get_state(self, observation):
        # Obtener datos del LiDAR y la posición del vehículo
        lidar_data = observation["lidar_point_cloud"]["point_cloud"]
        car_pose = np.array(observation["ego_vehicle_state"]["position"])

        # Filtrar puntos LiDAR vacíos (0,0,0) y calcular posiciones relativas
        relative_points = lidar_data[~np.all(lidar_data == [0, 0, 0], axis=1)] - car_pose

        # Si no hay puntos detectados, devolver solo la distancia y ángulo al objetivo
        if len(relative_points) == 0:
            distance_to_target = np.linalg.norm(self.target_position - car_pose)
            angle_to_target = np.arctan2(self.target_position[1] - car_pose[1], self.target_position[0] - car_pose[0])
            return (round(distance_to_target, 1), round(angle_to_target, 1))

        # Calcular distancias a obstáculos detectados y encontrar el más cercano
        distances = np.linalg.norm(relative_points, axis=1)
        min_distance = np.min(distances)
        closest_point = relative_points[np.argmin(distances)]
        
        # Calcular el ángulo al obstáculo más cercano
        angle_to_obstacle = np.arctan2(closest_point[1], closest_point[0])

        # Calcular distancia y ángulo al objetivo
        distance_to_target = np.linalg.norm(self.target_position - car_pose)
        angle_to_target = np.arctan2(self.target_position[1] - car_pose[1], self.target_position[0] - car_pose[0])

        # Estado: (distancia al objetivo, ángulo al objetivo, distancia al obstáculo más cercano, ángulo al obstáculo más cercano)
        return (round(distance_to_target, 1), round(angle_to_target, 1), round(min_distance, 1), round(angle_to_obstacle, 1))

    def choose_action(self, state):
        # Epsilon-greedy para seleccionar la acción
        if np.random.rand() < self.epsilon:
            # Explorar: Elegir una acción aleatoria
            return np.array([np.random.uniform(-2, 2), np.random.uniform(-1, 1)])

        # Explotar: Elegir la mejor acción conocida
        if state not in self.q_table:
            # Inicializar acción si el estado no está en la tabla Q
            self.q_table[state] = np.zeros((2,))
        return self.q_table[state]

    def act(self, observation):
        state = self.get_state(observation)
        action = self.choose_action(state)
        return action

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros((2,))

        # Calcular el valor Q futuro máximo
        max_future_q = np.max(self.q_table.get(next_state, np.zeros((2,))))

        # Actualizar la tabla Q con la fórmula de Q-learning
        current_q = self.q_table[state][0]
        self.q_table[state][0] = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)


def main(scenarios, headless, num_episodes=100, max_episode_steps=None):
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

    target_position = (23.25,100.0,0.0)

    env = gym.make(
        "smarts.env:hiway-v1",
        scenarios=scenarios,
        agent_interfaces={AGENT_ID: agent_interface},
        headless=headless,
    )

    env = ParkingAgent(env, target_position)
    

    for episode in episodes(n=num_episodes):
        agent = LearningAgent(target_position)
        observation, _ = env.reset()
        episode.record_scenario(env.unwrapped.scenario_log)

        terminated = False
        while not terminated:
            state = agent.get_state(observation)
            action = agent.act(observation)

            next_observation, reward, terminated, truncated, info = env.step((-action[0],-action[1]))
            next_state = agent.get_state(next_observation)

            agent.learn(state, action, reward, next_state)

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
        num_episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
    )
