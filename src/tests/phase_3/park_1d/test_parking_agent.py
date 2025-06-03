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

from q_table import q_table

AGENT_ID: Final[str] = "Agent"
MAX_ALIGN_STEPS = 19

def filtrate_lidar(lidar_data: np.ndarray, car_pose: np.ndarray, heading: float) -> np.ndarray:
    """
    Transforma los puntos LIDAR para que sean relativos al vehículo, con el índice 0 a 90° a la izquierda del agente.

    Args:
        lidar_data (np.ndarray): Datos del LIDAR en coordenadas absolutas.
        car_pose (np.ndarray): Posición actual del vehículo en coordenadas absolutas.
        heading (float): Ángulo de orientación del vehículo en radianes.

    Returns:
        np.ndarray: Datos LIDAR transformados en coordenadas relativas.
    """
    lidar_data_copy = np.copy(lidar_data)
    # Asignar 'inf' a los puntos inválidos (donde todo es [0, 0, 0])
    lidar_data_copy[np.all(lidar_data_copy == [0, 0, 0], axis=1)] = float('inf')

    # Calcular puntos relativos
    relative_points = lidar_data_copy - car_pose
    # relative_points = relative_points[::-1]

    # Convertir heading a grados
    heading_deg = np.degrees(heading)

    num_points = len(lidar_data_copy)
    lidar_resolution = 360 / num_points

    shift = int(round((heading_deg-90) / lidar_resolution))
    rotated_lidar = np.roll(relative_points, shift=shift, axis=0)

    return rotated_lidar


class LearningAgent:
    """Agente de Q-learning optimizado para aparcamiento, que utiliza LiDAR con velocidad angular fija."""

    def __init__(self):
        self.q_table = q_table # Tabla Q para almacenar los valores de estado-acción
        self.actions = [-1, 0, 1]  # Aceleraciones lineales posibles

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
        lidar_data = np.copy(observation["lidar_point_cloud"]["point_cloud"])
        filtrated_lidar = filtrate_lidar(observation["lidar_point_cloud"]["point_cloud"], np.array(observation["ego_vehicle_state"]["position"]), observation["ego_vehicle_state"]["heading"])
        # print("INSIDE:")
        # print(observation["lidar_point_cloud"]["point_cloud"])
        distances = np.linalg.norm(filtrated_lidar, axis=1)
        # choice = input("Option number: ")
        # print(observation["lidar_point_cloud"])
        lidar_resolution = 360 / len(distances)
        index_90 = int(round(90 / lidar_resolution))
        index_270 = int(round(270 / lidar_resolution))
        distance_90 = self.discretize(distances[index_90])
        # print(f"Distancia delante: {distance_90}")
        distance_270 = self.discretize(distances[index_270])
        # print(f"Distancia detras: {distance_270}")
        distance_difference = self.discretize(distance_270 - distance_90)

        velocity = observation['ego_vehicle_state']['speed']
        discretized_velocity = self.discretize(velocity, step=0.1, max_value=20)

        return (distance_difference, discretized_velocity)

    def choose_action(self, state):
        # Devolver la acción con el valor Q más alto en este estado
        return max(self.q_table[state], key=self.q_table[state].get)

    def act(self, observation):
        """Genera una acción basada en la observación del entorno."""
        state = self.get_state(observation)
        action = self.choose_action(state)
        # print(f"Accion elegida: {action}       En estado: {state}")
        return np.array([action, 0.0])

    def move_to_random_position(self, current_position, target_position, accelerate, steps, first_act):
        """Mueve el vehículo a una posición (target)."""

        distance = target_position - current_position
        action = 0

        # Determinar si avanzar o retroceder
        if accelerate == True:
            action = 10 if distance > 0 else -10

        # Paramos si estamos cerca o si llegamos a las maximas steps
        if abs(distance) < 0.25 or steps == MAX_ALIGN_STEPS:
            # print(f"finished, current pose: {current_position}")
            action = -first_act
                
        return np.array([action, 0.0])



def main(scenarios, headless, num_episodes=200, max_episode_steps=None):
    agent_interface = AgentInterface(
        action=ActionSpaceType.Direct,
        # max_episode_steps=max_episode_steps,
        max_episode_steps=200,
        neighborhood_vehicle_states=True,
        # waypoint_paths=True,
        # road_waypoints=True,
        drivable_area_grid_map=True,
        occupancy_grid_map=True,
        top_down_rgb=True,
        lidar_point_cloud=True,
        accelerometer=True,
        lane_positions=True,
        signals=True,
    )

    env = gym.make(
        "smarts.env:hiway-v1",
        scenarios=scenarios,
        agent_interfaces={AGENT_ID: agent_interface},
        headless=headless,
    )

    env = ParkingAgent(env)
    agent = LearningAgent()
    values = [-1, 0, 1, 2]
    n = 0

    for episode in episodes(n=num_episodes):
        observation, _ = env.reset()
        episode.record_scenario(env.unwrapped.scenario_log)

        # np.random.seed(int(time.time()))
        if n >= len(values):
            n = 0
        offset = values[n]
        actual_pose = observation["ego_vehicle_state"]["position"][0]
        target =  actual_pose + offset

        # print(f"Moving from pose: {actual_pose} to pose: {target}")

        terminated = False
        moved = False
        
        accelerate = True
        first_action = np.array([0.0, 0.0])
        n_steps = 0

        # t_reward = 0
        while not terminated:
            # Mover a posicion aleatoria
            if not moved:
                # Indice 0 es el que se usa en nuestro escenario
                action = agent.move_to_random_position(observation["ego_vehicle_state"]["position"][0], target, accelerate, n_steps, first_action[0])
                accelerate = False
                
                if action[0] + first_action[0] == 0:
                    moved = True

                if n_steps == 0:                    
                    first_action = action
                    
                observation, _, terminated, _, _ = env.step((action[0],action[1]))
                n_steps = n_steps + 1
                # print(observation['ego_vehicle_state']['speed'])
            
            # Tenemos que asegurarnos que SIEMPRE gastamos MAX_ALIGN_STEPS steps, así no modificamos el entrenamiento
            elif n_steps <= MAX_ALIGN_STEPS:
                # print(observation['ego_vehicle_state']['speed'])
                observation, _, terminated, _, _ = env.step((0.0,0.0))
                n_steps = n_steps + 1

            # Nos quedan TOTAL_STEPS-MAX_ALIGN_STEPS para el entrenamiento, SIEMPRE las mismas
            else:
                
                state = agent.get_state(observation)
                print(f"{state[0]}")
                action = agent.act(observation)

                next_observation, reward, terminated, truncated, info = env.step((action[0],action[1]))

                observation = next_observation
                episode.record_step(observation, reward, terminated, truncated, info)
        n = n + 1
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
        num_episodes=50,
        max_episode_steps=200,
    )