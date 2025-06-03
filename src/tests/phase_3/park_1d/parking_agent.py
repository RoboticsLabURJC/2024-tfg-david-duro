from typing import Any, Tuple
import numpy as np
import gymnasium as gym

class ParkingAgent(gym.Wrapper):
    """Un agente adaptado para estacionamiento utilizando medidas específicas del LIDAR."""

    def __init__(self, env: gym.Env):
        """
        Args:
            env (gym.Env): Entorno de SMARTS para un solo agente.
        """
        super(ParkingAgent, self).__init__(env)

        agent_ids = list(env.agent_interfaces.keys())
        assert (
            len(agent_ids) == 1
        ), f"Expected env to have a single agent, but got {len(agent_ids)} agents."
        self._agent_id = agent_ids[0]

        if self.observation_space:
            self.observation_space = self.observation_space[self._agent_id]
        if self.action_space:
            self.action_space = self.action_space[self._agent_id]

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Any]:
        """Realiza un paso en el entorno SMARTS y calcula la recompensa de estacionamiento.

        Args:
            action (Any): Acción a realizar por el agente.

        Returns:
            Tuple[Any, float, bool, bool, Any]: Observación, recompensa, indicadores de fin de episodio y datos adicionales.
        """
        obs, _, terminated, truncated, info = self.env.step({self._agent_id: action})
        
        agent_obs = obs[self._agent_id]
        
        filtrated_lidar = self.filtrate_lidar(agent_obs["lidar_point_cloud"]["point_cloud"], np.array(agent_obs["ego_vehicle_state"]["position"]), agent_obs["ego_vehicle_state"]["heading"])
        # print(agent_obs["lidar_point_cloud"]["point_cloud"])
        # print("---------------------R------------------------")
        # print(agent_obs["lidar_point_cloud"]["point_cloud"][len(agent_obs["lidar_point_cloud"]["point_cloud"]) - 1])
        car_pose = np.array(agent_obs["ego_vehicle_state"]["position"])
        heading = agent_obs["ego_vehicle_state"]["heading"]
        car_speed = agent_obs['ego_vehicle_state']['speed']

        reward = self._compute_parking_reward(filtrated_lidar, car_speed)

        return (
            agent_obs,
            reward,
            terminated[self._agent_id],
            truncated[self._agent_id],
            info[self._agent_id],
        )

    def reset(self, *, seed=None, options=None) -> Tuple[Any, Any]:
        """Reinicia el entorno de SMARTS y devuelve la observación inicial.

        Returns:
            Tuple[Any, Any]: Observación y datos adicionales.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        return obs[self._agent_id], info[self._agent_id]

    def _compute_parking_reward(self, lidar_data: np.ndarray, speed) -> float:
        """Calcula la recompensa basada en las medidas de LIDAR a 90° y 270°.

        Args:
            lidar_data (np.ndarray): Datos del punto LIDAR alrededor del vehículo.
            car_pose (np.ndarray): Posición actual del agente (coordenadas absolutas).

        Returns:
            float: Recompensa calculada.
        """

        distances = np.linalg.norm(lidar_data, axis=1)
        lidar_resolution = 360 / len(distances)
        index_90 = int(round(90 / lidar_resolution))
        index_270 = int(round(270 / lidar_resolution))
        distance_90 = distances[index_90]
        distance_270 = distances[index_270]

        if np.isinf(distance_90) or np.isinf(distance_270):
            return 0
        
        # Calcula la diferencia absoluta entre las dos distancias.
        distance_difference = abs(distance_90 - distance_270)

        # Soluciona 1/0 y recompensas infinitas
        if distance_difference < 0.1:
            distance_difference = 0.1
        
        # Centrado y estatico tiene que ser mayor que recompensa estandar maxima
        if distance_difference <= 0.17 and abs(speed) < 0.001:
            reward = 200
        # Prevee choque y penaliza
        elif distance_difference > 4.5 or abs(speed) > 12:
            reward = -10
        # Recompensa estandar maxima = 100 (1/0.01)
        else:
            reward = (1/distance_difference)

        # print(f"Recompensa: {reward}")
        return reward
    
    def filtrate_lidar(self, lidar_data: np.ndarray, car_pose: np.ndarray, heading: float) -> np.ndarray:
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
        # Aplicar el desplazamiento circular
        rotated_lidar = np.roll(relative_points, shift=shift, axis=0)

        return rotated_lidar




