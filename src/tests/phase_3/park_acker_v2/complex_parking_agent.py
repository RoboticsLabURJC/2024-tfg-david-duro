from typing import Any, Tuple
import numpy as np
import gymnasium as gym

TARGET_HEADING = -np.pi/2
MAX_ALIGN_STEPS = 19
MAX_STEPS = 500
# MAX_DIST = 8.5
MAX_DIST = 10

class CParkingAgent(gym.Wrapper):
    """Un agente adaptado para estacionamiento utilizando medidas específicas del LIDAR."""

    def __init__(self, env: gym.Env):
        """
        Args:
            env (gym.Env): Entorno de SMARTS para un solo agente.
        """
        super(CParkingAgent, self).__init__(env)
        self.step_number = 0
        self.last_orientation = 0
        self.min_orient = 0.7
        self.last_target_distance = 0
        self. last_target_pose = (0,0)

        agent_ids = list(env.agent_interfaces.keys())
        assert (
            len(agent_ids) == 1
        ), f"Expected env to have a single agent, but got {len(agent_ids)} agents."
        self._agent_id = agent_ids[0]

        if self.observation_space:
            self.observation_space = self.observation_space[self._agent_id]
        if self.action_space:
            self.action_space = self.action_space[self._agent_id]

    def step(self, action: Any, target: np.ndarray) -> Tuple[Any, float, bool, bool, Any]:
        """Performs a step in the SMARTS environment and calculates the parking reward.

        Args:
            action (Any): Action taken by the agent.
            target (np.ndarray): Target position as a 3-element array (x, y, theta).

        Returns:
            Tuple[Any, float, bool, bool, Any]: Observation, reward, episode end flags, and additional info.
        """
        obs, _, terminated, truncated, info = self.env.step({self._agent_id: action})
        
        agent_obs = obs[self._agent_id]
        
        filtrated_lidar = self.filtrate_lidar(agent_obs["lidar_point_cloud"]["point_cloud"], np.array(agent_obs["ego_vehicle_state"]["position"]), agent_obs["ego_vehicle_state"]["heading"])
        car_pose = np.array(agent_obs["ego_vehicle_state"]["position"])
        heading = agent_obs["ego_vehicle_state"]["heading"]
        car_speed = agent_obs['ego_vehicle_state']['speed']

        reward = self._compute_parking_reward(car_pose, heading, car_speed, target , TARGET_HEADING, filtrated_lidar)

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
        self.min_orient = 0.7
        self.last_target_distance = 0
        self.last_target_pose = (0,0)
        obs, info = self.env.reset(seed=seed, options=options)
        return obs[self._agent_id], info[self._agent_id]

    def _compute_parking_reward(
        self,
        car_pose: np.ndarray,
        car_orient: float,
        speed: float,
        target_pose: np.ndarray,
        target_orient: float,
        lidar_data: np.ndarray,
    ) -> float:

        dist_to_target = np.linalg.norm(target_pose)
        horizontal_dist = abs(target_pose[1])
        vertical_dist = abs(target_pose[0])
        orient_diff = np.abs(np.arctan2(np.sin(car_orient - target_orient), np.cos(car_orient - target_orient)))
        reward = 0

        # Recompensa por distancia
        distance_reward = (1 / (1 + np.exp(2.5 * (dist_to_target - 2))))
        if vertical_dist > 6:
            distance_reward -= 0.1
        reward += np.clip(distance_reward, 0, 2)

        # Recompensa por orientación (progresiva)
        orientation_weight = np.exp(-2 * horizontal_dist)  # Más importante cuando está cerca
        orientation_reward = orientation_weight * (3.5 * np.exp(-2.5 * orient_diff) - 0.75)
        orientation_reward = np.clip(orientation_reward, -0.5, 3)

        # Penalización si está estancado en orientación (fomenta corregir)
        if self.step_number > 10 and abs(self.last_orientation - orient_diff) < 0.01:
            orientation_reward -= 0.3
            # print(f"[ORIENT STUCK] ΔOrient: {abs(self.last_orientation - orient_diff):.4f}")

        # Recompensa por progresar en orientación (fomenta mejora)
        if horizontal_dist < 0.3:
            if orient_diff < self.min_orient:
                self.min_orient = orient_diff
                orientation_reward += 0.3
                # print(f"[ORIENT IMPROVED] Reward += 0.3")
        
        # Recompensa acercamiento marcha atrás o penalizacion marcha alante
        if speed < -0.1 and horizontal_dist < self.last_target_pose[1] - 0.05 and self.step_number > 10:
            orientation_reward += 0.3

        elif speed > 0.1 and horizontal_dist < self.last_target_pose[1] - 0.05 and horizontal_dist > 0.5 and self.step_number > 10:
            orientation_reward -= 0.5

        reward += orientation_reward

        # Bonus progresivo por objetivo final
        # perfect_bonus = 10 * np.exp(-10 * horizontal_dist**2) * np.exp(-10 * vertical_dist**2) * np.exp(-5 * orient_diff**2)
        perfect_bonus = 10 * np.exp(-7.5 * horizontal_dist**2) * np.exp(-10 * vertical_dist**2) * np.exp(-5 * orient_diff**2)
        if abs(speed) < 0.15:
            reward += perfect_bonus
            if perfect_bonus > 1.0:
                print(f"[BONUS FINAL] +{perfect_bonus:.2f}")

        # Penalización por choque
        min_lidar_dist = np.min(np.linalg.norm(lidar_data, axis=1)) if len(lidar_data) > 0 else np.inf
        if min_lidar_dist < 0.1:
            print(f"[COLLISION] Lidar min dist: {min_lidar_dist}")
            reward -= 5

        # Penalización por velocidad
        if abs(speed) > 4:
            reward -= 0.5

        # Penalización por salir de la zona útil
        if dist_to_target >= MAX_DIST or target_pose[1] < -0.8:
            reward -= 5
            print(f"[OUT OF RANGE] Vertical: {vertical_dist:.2f}, Horizontal: {target_pose[1]:.2f}")

        # Penalización por intentar aparcar de frente
        if car_orient < target_orient and orient_diff > 0.3:
            print(f"[FRONTAL PARKING] Car orient: {car_orient:.2f}, Target: {target_orient:.2f}")
            reward -= 5

        # Penalización fuerte por orientación absurda
        if orient_diff > ((5 * np.pi) / 12):  # > 75º
            reward -= 5
            print("[BAD ORIENTATION] Exceeded max allowed angle")

        # Penalización por paso máximo
        if self.step_number >= MAX_STEPS:
            reward -= 100

        # Guardar orientación para análisis en siguientes pasos
        self.last_orientation = orient_diff
        self.last_target_distance = dist_to_target
        self.last_target_pose = (vertical_dist, horizontal_dist)

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




