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
        lidar_data = agent_obs["lidar_point_cloud"]["point_cloud"]
        car_pose = np.array(agent_obs["ego_vehicle_state"]["position"])

        reward = self._compute_parking_reward(lidar_data, car_pose)

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

    def _compute_parking_reward(self, lidar_data: np.ndarray, car_pose: np.ndarray) -> float:
        """Calcula la recompensa basada en las medidas de LIDAR a 90° y 270°.

        Args:
            lidar_data (np.ndarray): Datos del punto LIDAR alrededor del vehículo.
            car_pose (np.ndarray): Posición actual del agente (coordenadas absolutas).

        Returns:
            float: Recompensa calculada.
        """
        lidar_length = len(lidar_data)
        index_90 = lidar_length // 4
        index_270 = (3 * lidar_length) // 4 

        # Asignar 'inf' a los puntos donde no hay obstáculos ([0, 0, 0]).
        lidar_data[np.all(lidar_data == [0, 0, 0], axis=1)] = float('inf')

        relative_lidar = lidar_data - car_pose
        distances = np.linalg.norm(relative_lidar, axis=1)

        distance_90 = distances[index_90]
        distance_270 = distances[index_270]

        if np.isinf(distance_90) or np.isinf(distance_270):
            return -10.0

        # Calcula la diferencia absoluta entre las dos distancias.
        distance_difference = abs(distance_90 - distance_270)

        # Rangos de recompensa según la diferencia de distancias.
        if distance_difference < 0.5:
            reward = 5.0  # Perfecto equilibrio.
        # elif distance_difference < 1.0:
        #     reward = -1.0  # Ligera penalización.
        # elif distance_difference < 1.5:
        #     reward = -3.0  # Penalización moderada.
        else:
            reward = (1/distance_difference)*2

        return reward

