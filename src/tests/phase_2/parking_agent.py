from typing import Any, Tuple
import numpy as np
import gymnasium as gym

class ParkingAgent(gym.Wrapper):
    """Un agente adaptado para estacionamiento utilizando recompensas
    personalizadas basadas en medidas de LIDAR, con obstáculos en coordenadas relativas."""

    def __init__(self, env: gym.Env, target_position: Tuple[float, float, float]):
        """
        Args:
            env (gym.Env): Entorno de SMARTS para un solo agente.
            target_position (Tuple[float, float, float]): Posición objetivo (x, y, z) para el estacionamiento.
        """
        super(ParkingAgent, self).__init__(env)

        agent_ids = list(env.agent_interfaces.keys())
        assert (
            len(agent_ids) == 1
        ), f"Expected env to have a single agent, but got {len(agent_ids)} agents."
        self._agent_id = agent_ids[0]
        self.target_position = np.array(target_position)

        if self.observation_space:
            self.observation_space = self.observation_space[self._agent_id]
        if self.action_space:
            self.action_space = self.action_space[self._agent_id]

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Any]:
        """Realiza un paso en el entorno SMARTS y calcula la recompensa de estacionamiento.

        Args:
            action (Any): Acción a realizar por el agente

        Returns:
            Tuple[Any, float, bool, bool, Any]: Observación, recompensa, indicadores de fin de episodio y datos adicionales
        """
        obs, _, terminated, truncated, info = self.env.step({self._agent_id: action})
        
        agent_obs = obs[self._agent_id]
        lidar_data = agent_obs["lidar_point_cloud"]["point_cloud"]
        agent_position = np.array(agent_obs["ego_vehicle_state"]["position"])

        reward = self._compute_parking_reward(lidar_data, agent_position)

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
            Tuple[Any, Any]: Observación y datos adicionales
        """
        obs, info = self.env.reset(seed=seed, options=options)
        return obs[self._agent_id], info[self._agent_id]

    def closest_obstacle_warning(self, measurements: np.ndarray, car_pose: np.ndarray) -> Tuple[float, float]:
        """Convierte coordenadas absolutas de obstáculos a relativas y calcula la distancia y ángulo más cercanos.

        Args:
            measurements (np.ndarray): Coordenadas absolutas de obstáculos (LIDAR).
            car_pose (np.ndarray): Posición actual del agente.

        Returns:
            Tuple[float, float]: Distancia mínima y ángulo relativo al obstáculo más cercano.
        """
        if np.any(measurements != 0):
            measurements = measurements[~np.all(measurements == [0, 0, 0], axis=1)] - car_pose
            distances = np.linalg.norm(measurements, axis=1)
            min_distance = np.min(distances)
            
            closest_point = measurements[np.argmin(distances)]
            angle_to_obstacle = np.arctan2(closest_point[1], closest_point[0])
            
            return min_distance, angle_to_obstacle
        return float('inf'), 0.0

    def _compute_parking_reward(self, lidar_data: np.ndarray, agent_position: np.ndarray) -> float:
        """Calcula una recompensa basada en la distancia a la posición objetivo y la proximidad a obstáculos.

        Args:
            lidar_data (np.ndarray): Datos del punto LIDAR alrededor del vehículo
            agent_position (np.ndarray): Posición actual del agente

        Returns:
            float: Recompensa calculada
        """
        distance_to_target = np.linalg.norm(agent_position - self.target_position)
        reward = -distance_to_target

        min_obstacle_distance, angle_to_obstacle = self.closest_obstacle_warning(lidar_data, agent_position)
        
        if min_obstacle_distance < 0.4:
            reward -= 6
        elif min_obstacle_distance < 0.2:
            reward -= 10 

        return reward
