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
from smarts.env.gymnasium.wrappers.complex_parking_agent import CParkingAgent

SMARTS_REPO_PATH = Path(__file__).parents[1].absolute()
sys.path.insert(0, str(SMARTS_REPO_PATH))
from examples.tools.argument_parser import minimal_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, ActionSpaceType, AgentType
from smarts.core.utils.episodes import episodes
from smarts.sstudio.scenario_construction import build_scenarios
from smarts.core.scenario import Scenario

import time
import random

from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import math

MAX_ALIGN_STEPS = 19
STEPS = 500
MAX_DIST = 710

AGENT_ID: Final[str] = "Agent"

TARGET_HEADING = -np.pi/2
dir_path = "data_log/park_final/v1/"
# dir_path = ""
model_path = "/home/duro/SMARTS/examples/"+ dir_path +"dqn_model.pth"
# ROAD: -pi/2
# DEST: [28.16 96.8   0.  ]

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

    # Reordenar de (y, x, z) a (x, y, z)
    # lidar_data_copy = lidar_data_copy[:, [1, 0, 2]]
    
    # Calcular puntos relativos en el nuevo formato
    relative_points = car_pose - lidar_data_copy# - car_pose
    relative_points = relative_points[:, [1, 0, 2]]
    
    # Matriz de rotación en el sistema dextrógiro
    rotation_matrix = np.array([
        [ np.cos(heading), np.sin(heading), 0],  # x'
        [-np.sin(heading), np.cos(heading), 0],  # y'
        [0,                0,               1]  # z no cambia
    ])

    # Aplicar la transformación de rotación
    rotated_points = relative_points @ rotation_matrix.T

    # Convertir heading a grados
    heading_deg = np.degrees(heading)

    num_points = len(lidar_data_copy)
    lidar_resolution = 360 / num_points

    shift = int(round((heading_deg - 90) / lidar_resolution))
    # Aplicar el desplazamiento circular
    rotated_lidar = np.roll(rotated_points, shift=shift, axis=0)

    return rotated_lidar

class SuccessTracker:
    def __init__(self):
        """Inicializa la clase con una lista vacía de puntuaciones."""
        self.scores = []

    def add_score(self, score):
        """
        Añade una nueva puntuación a la lista de puntuaciones.

        Parámetros:
            score (float): La puntuación a añadir.
        """
        self.scores.append(score)

    def success_rate(self, threshold):
        """
        Calcula el porcentaje de éxito basado en un umbral.

        Parámetros:
            threshold (float): El umbral para considerar una puntuación como éxito.

        Retorna:
            float: El porcentaje de éxito (entre 0 y 100).
        """
        if not self.scores:
            return 0.0  # Si no hay puntuaciones, el porcentaje de éxito es 0

        # Contar el número de puntuaciones que superan el umbral
        success_count = sum(1 for score in self.scores if score >= threshold)
        
        # Calcular el porcentaje de éxito
        return (success_count / len(self.scores)) * 100

    def reset(self):
        """Reinicia la lista de puntuaciones."""
        self.scores = []


class Desalignment:
    def __init__(self, env, max_align_steps=MAX_ALIGN_STEPS):
        self.env = env
        self.max_align_steps = max_align_steps

    def reset(self, observation, rotate=False):
        """Reinicia los parámetros de desalineación."""
        self.moved = False
        self.rotate = rotate
        self.n_steps = 0
        self.accelerate = True
        self.random_offset = np.random.choice([-2, -1.75, -1.5, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.5, 1.75, 2])
        # self.random_offset = np.random.choice([-2, 0, 2])
        self.random_rotation = np.random.choice([-0.005, -0.002, 0, 0.002, 0.005])
        self.target = observation["ego_vehicle_state"]["position"][0] + self.random_offset
        self.steering = self.random_rotation
        self.speed = 0
    
    def move_to_random_position(self, current_position, target_position, accelerate, steps):
        """Mueve el vehículo a una posición (target)."""
        throttle = 0
        brake = 0
        steering = 0
        

        distance = target_position - current_position
        if abs(self.speed ) < 3:
            throttle = 1 if distance > 0 else -1
        else:
            throttle = 0

        if abs(distance) < 0.1:
            # print(f"finished, current pose: {current_position}, target pose: {target_position}")
            self.accelerate = False
        
        if not self.accelerate and abs(self.speed) > 0.2:
            brake = 1
            steering = self.steering
            
            
        if steps == self.max_align_steps:
            self.moved = True
            throttle = 0
            brake = 0
            steering = 0
                
        return np.array([throttle, brake, steering], dtype=np.float32)

    def run(self, observation, parking_target):
        """Mueve el vehículo a una posición aleatoria."""
        self.speed = observation['ego_vehicle_state']['speed']
        if not self.moved:
            action = self.move_to_random_position(
                observation["ego_vehicle_state"]["position"][0], self.target, self.accelerate, self.n_steps
            )
            observation, _, terminated, _, _ = self.env.step(action, parking_target)
            self.n_steps += 1
            return observation, terminated

        elif self.n_steps <= self.max_align_steps:
            steering = 0.0
            observation, _, terminated, _, _ = self.env.step(np.array([0.0, 0.0, steering], dtype=np.float32), parking_target)
            self.n_steps += 1
            return observation, terminated

        else:
            return observation, False

    def is_desaligned(self):
        """Devuelve True si la desalineación está en progreso, False si ha terminado."""
        return self.n_steps <= self.max_align_steps

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

        self.layer_norm1 = nn.LayerNorm(128)  # Reemplaza BatchNorm1d
        self.layer_norm2 = nn.LayerNorm(64)

    def forward(self, x):
        x = torch.relu(self.layer_norm1(self.fc1(x)))  # Ahora usa LayerNorm
        x = torch.relu(self.layer_norm2(self.fc2(x)))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, model_path):
        # self.actions = [(0.0, 0.7, 0.0),(0.0, 0.0, -1.0),(-1.0, 0.0, 0.0),(0.0, 0.0, 0.0),(1.0, 0.0, 0.0),(0.0, 0.0, 1.0)]
        # self.actions = [(0.0, 0.7, 0.0),(0.0, 0.0, -1.0),(-1.0, 0.0, 0.0),(0.0, 0.0, 0.0),(1.0, 0.0, 0.0),(0.0, 0.0, 1.0),
        #  (1.0, 0.0, 1.0), (1.0, 0.0, -1.0), (-1.0, 0.0, 1.0), (-1.0, 0.0, -1.0)]
        # self.actions = [(0.0, 0.7, 0.0),(0.0, 0.0, -1.0),(-1.0, 0.0, 0.0),(1.0, 0.0, 0.0),(0.0, 0.0, 1.0),
        #  (-0.5, 0.0, 1.0), (-0.5, 0.0, -1.0)]
        self.actions = [(0.0, 0.7, 0.0),(0.0, 0.0, -1.0),(-1.0, 0.0, 0.0),(1.0, 0.0, 0.0),(0.0, 0.0, 1.0),
         (0.5, 0.0, 1.0), (0.5, 0.0, -1.0), (-0.5, 0.0, 1.0), (-0.5, 0.0, -1.0)]


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, len(self.actions)).to(self.device)
        
        # Cargar el modelo entrenado
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # Poner en modo evaluación

        # DEBUG
        self.reward = 0
        self.steps = 0

        self.action_space = self.actions

        self.init_pose = np.array([0, 0, 0])
        self.parking_target_pose = np.array([0, 0, 0])

    def act(self, state):
        """Selecciona una acción usando el modelo entrenado."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        
        action_index = q_values.argmax().item()  # Escoge la acción con el mayor valor Q
        return self.action_space[action_index]


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
        discretized = round(value / step) * step
        # Truncar a un decimal
        return math.floor(discretized * 10) / 10

    def get_relative_coordinates(self, position, heading):
        """
        Transforma una posición global a coordenadas relativas usando la orientación (heading) en 3D.
        
        Parámetros:
            position (array-like): Coordenadas [x, y, z] en el sistema global.
            heading (float): Ángulo de orientación en radianes.

        Retorna:
            np.ndarray: Coordenadas [x', y', z'] en el sistema relativo.
        """
        init_position = np.array(self.init_pose)

        # Calcular el desplazamiento en coordenadas globales
        delta_position = position - init_position
        delta_position = np.array([delta_position[0], -delta_position[1], delta_position[2]])

        # Los puntos absolutos no están en funcion de 0º sino de la orientacion de la carretera

        rotation_matrix = np.array([
            [np.cos(heading), np.sin(heading), 0],
            [-np.sin(heading),  np.cos(heading), 0],
            [0,                0,                1]  # Z no cambia
        ])
        # # Aplicar la transformación
        pose_relative = rotation_matrix @ delta_position

        return delta_position

    def get_state(self, observation, target_pose):
        """Extrae y discretiza el estado basado en la posición, orientación, velocidad y LiDAR del vehículo."""
        
        # Extraer información relevante
        car_position = np.array(observation["ego_vehicle_state"]["position"])
        car_heading = observation["ego_vehicle_state"]["heading"]
        car_speed = observation["ego_vehicle_state"]["speed"]
        
        # Calcular la posición relativa del vehículo con respecto a init_pose
        car_pose_relative = self.get_relative_coordinates(car_position, TARGET_HEADING)
        
        # Sumar la posición relativa del vehículo con target_pose (relativo a init_pose)
        self.parking_target_pose = target_pose - car_pose_relative
        # print(f"TARGET RELATIVE: {self.parking_target_pose}, TARGET INIT: {target_pose}, CAR_POSE: {car_position}")
        
        # Extraer componentes x (horizontal) e y (vertical)
        x_rel = self.parking_target_pose[0]
        y_rel = self.parking_target_pose[1]

        # Discretizar cada componente por separado
        discretized_x = self.discretize(x_rel, step=0.2, max_value=MAX_DIST)
        discretized_y = self.discretize(y_rel, step=0.2, max_value=MAX_DIST)

        # Diferencia de orientación ajustada a [-pi, pi]
        heading_error = np.arctan2(
            np.sin(TARGET_HEADING - car_heading), 
            np.cos(TARGET_HEADING - car_heading)
        )
        
        # Filtrar datos del LiDAR
        filtered_lidar = filtrate_lidar(
            observation["lidar_point_cloud"]["point_cloud"], 
            car_position, 
            car_heading
        )
        
        distances = np.linalg.norm(filtered_lidar, axis=1)
        distances = np.nan_to_num(distances, nan=0.0, posinf=1e6, neginf=-1e6)  # Reemplazar NaN e Inf
        
        # Discretizar velocidad para mejor aprendizaje
        discretized_speed = self.discretize(car_speed, step=0.1, max_value=5)

        # Encontrar la distancia mínima válida en el array 'distances'
        distances[distances == 0] = 1e6
        min_index = np.argmin(distances)
        min_distance = distances[min_index]

        discretized_min_distance = self.discretize(min_distance, 0.1, 1.5)

        # Devolver estado asegurando que todos los valores sean finitos
        return (
            discretized_x,
            discretized_y,
            self.discretize(heading_error, step=0.1, max_value=np.pi),
            discretized_speed,
            discretized_min_distance
        )

    def find_closest_corners(self, observation, eps=3, min_samples=10):
        """
        Encuentra las esquinas más cercanas de dos vehículos que delimitan un hueco de aparcamiento a partir de un point cloud.
        
        :param point_cloud: Nube de puntos del escenario (Nx3 numpy array)
        :param eps: Parámetro de distancia para el algoritmo DBSCAN (tolerancia en la agrupación de puntos).
        :param min_samples: Número mínimo de puntos para que un conjunto sea considerado un clúster (DBSCAN).
        
        :return: Punto medio entre las esquinas más cercanas (numpy array con las coordenadas [x, y, z]).
        """

        pose = np.array(observation["ego_vehicle_state"]["position"])
        self.init_pose = pose
        filtrated_lidar = filtrate_lidar(observation["lidar_point_cloud"]["point_cloud"], pose, observation["ego_vehicle_state"]["heading"])

        # 1. Filtrar los puntos que representan ruido o valores infinitos
        filtrated_lidar = filtrated_lidar[~np.isnan(filtrated_lidar).any(axis=1)]  # Elimina NaNs
        filtrated_lidar = filtrated_lidar[~np.isinf(filtrated_lidar).any(axis=1)]  # Elimina infs
        
        # 2. Aplicar DBSCAN para dividir el point cloud en dos grupos (vehículos)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(filtrated_lidar)
        
        
        # Obtener las etiquetas de los clusters y filtramos solo aquellos que tienen más de 10 puntos
        labels = clustering.labels_

        # plot_clusters(point_cloud, labels)
        
        # Asegurarnos de que hay al menos dos clústeres
        unique_labels = np.unique(labels)
        if len(unique_labels) != 2:
            print("No se detectaron dos vehículos en el point cloud.")
            return None
            
        
        # Dividir los puntos en dos grupos (vehículos)
        group_1 = filtrated_lidar[labels == unique_labels[0]]
        group_2 = filtrated_lidar[labels == unique_labels[1]]
        
        # 3. Encontrar las esquinas más cercanas entre los dos vehículos
        closest_distance = np.inf
        closest_pair = None
        
        for corner_1 in group_1:
            for corner_2 in group_2:
                distance = np.linalg.norm(corner_1 - corner_2)  # Distancia euclidiana
                if distance < closest_distance:
                    closest_distance = distance
                    closest_pair = (corner_1, corner_2)
        
        # 4. Calcular el punto medio entre las dos esquinas más cercanas
        if closest_pair is None:
            raise ValueError("No se pudo encontrar un par de esquinas cercanas.")
        
        corner_1, corner_2 = closest_pair
        midpoint = (corner_1 + corner_2) / 2
        midpoint[2] = 0
        midpoint[1] = midpoint[1] + 0.5 # Le sumamos 1/2 ancho del coche para centrar aparcamiento
        
        return midpoint



def main(scenarios, headless, num_episodes=300, max_episode_steps=None):
    agent_interface = AgentInterface(
        action=ActionSpaceType.Continuous,
        # max_episode_steps=max_episode_steps,
        max_episode_steps=STEPS,
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
    # env.fast_mode = True

    env = CParkingAgent(env)
    agent = DQNAgent(state_size=5, action_size=9, model_path=model_path)
    n_ep = 0

    desalignment = Desalignment(env, MAX_ALIGN_STEPS)
    success_tracker = SuccessTracker()

    print(f"El modelo se ejecutará en: {agent.device}")
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    
    for episode in episodes(n=num_episodes):
        agent.reward = 0
        observation, _ = env.reset()
        episode.record_scenario(env.unwrapped.scenario_log)

        # Reiniciar la desalineación
        desalignment.reset(observation, True)

        terminated = False
        agent.steps = 0

        parking_target = agent.find_closest_corners(observation)
        if parking_target is None:
            terminated = True
        # print(f"El target se encuentra en: {parking_target}")
        while not terminated:
            env.step_number = agent.steps
            # Mover a posicion aleatoria
            if desalignment.is_desaligned():
                observation, terminated = desalignment.run(observation, parking_target)
                continue

            state = agent.get_state(observation, parking_target)
            action = agent.act(state)

            next_observation, reward, terminated, truncated, info = env.step((action[0],action[1], action[2]), agent.parking_target_pose)

            observation = next_observation
            episode.record_step(observation, reward, terminated, truncated, info)
            # if abs(state[0]) + 0.1 > 7.5 or (abs(state[1])) + 0.1 > ((5 * np.pi) / 12) or reward > 200:
            #     terminated = True
            #     break
            # Terminar el programa si hay problemas o exito
            agent.reward += reward
            agent.steps += 1
            if reward <= -3 or reward > 25:
                terminated = True
                print("TERMINADO!")
                break
            if env.last_orientation < 0.1 and env.last_target_distance < 0.3 and abs(agent.parking_target_pose[1] < 0.1) and agent.steps > 150 and agent.reward > 200:
                print("-----[PARKING FINALIZADO]-----")
                agent.reward += 500
                break
        success_tracker.add_score(agent.reward)
    env.close()

    success_rate = success_tracker.success_rate(400)
    print(f"El porcentaje de exito ha sido: {success_rate}")


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
        max_episode_steps=STEPS,
    )
