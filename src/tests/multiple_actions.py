import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Final
import gymnasium as gym
import numpy as np
from scipy.integrate import odeint

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

SMARTS_REPO_PATH = Path(__file__).parents[1].absolute()
sys.path.insert(0, str(SMARTS_REPO_PATH))

from examples.tools.argument_parser import minimal_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, ActionSpaceType
from smarts.core.utils.episodes import episodes
from smarts.env.gymnasium.wrappers.single_agent import SingleAgent
from smarts.sstudio.scenario_construction import build_scenarios

from smarts.core.controllers.actuator_dynamic_controller import ActuatorDynamicControllerState

AGENT_ID: Final[str] = "Agent"

class PIDController:
    """Simple PID controller"""
    def __init__(self, kp, ki, kd, setpoint=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.last_error = 0.0
        self.integral = 0.0

    def update(self, current_value, dt):
        error = self.setpoint - current_value
        derivative = (error - self.last_error) / dt
        self.integral += error * dt
        self.last_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class KeepLaneAgent(Agent):
    def __init__(self):
        self.state = ActuatorDynamicControllerState()

        self.throttle_pid = PIDController(kp=0.1, ki=0.01, kd=0.01, setpoint=0.0)
        self.brake_pid = PIDController(kp=0.1, ki=0.01, kd=0.01, setpoint=0.0)
        self.steering_pid = PIDController(kp=0.1, ki=0.01, kd=0.01, setpoint=0.0)

        self.current_throttle = 0.0
        self.current_brake = 0.0
        self.current_steering_rate = 0.0

    def act(self, obs, **kwargs):
        # Random action
        action = np.random.choice(['accelerate', 'slow_down', 'turn_left', 'turn_right'])

        dt = 0.1

        # Random values for each action
        if action == 'accelerate':
            self.throttle_pid.setpoint = np.random.uniform(0.5, 1.0)
            self.brake_pid.setpoint = 0.0
        elif action == 'slow_down':
            self.throttle_pid.setpoint = 0.0
            self.brake_pid.setpoint = np.random.uniform(0.5, 1.0)
        elif action == 'turn_left':
            self.steering_pid.setpoint = np.random.uniform(-1.0, -0.5)
        elif action == 'turn_right':
            self.steering_pid.setpoint = np.random.uniform(0.5, 1.0)

        # Smoothing values
        self.current_throttle = self.throttle_pid.update(self.current_throttle, dt)
        self.current_brake = self.brake_pid.update(self.current_brake, dt)
        self.current_steering_rate = self.steering_pid.update(self.current_steering_rate, dt)

        # Limiting values
        self.current_throttle = np.clip(self.current_throttle, 0.0, 1.0)
        self.current_brake = np.clip(self.current_brake, 0.0, 1.0)
        self.current_steering_rate = np.clip(self.current_steering_rate, -1.0, 1.0)

        return self.current_throttle, self.current_brake, self.current_steering_rate

def main(scenarios, headless, num_episodes, max_episode_steps=None):
    # Configuration to use ActuatorDynamic
    agent_interface = AgentInterface(
        action=ActionSpaceType.ActuatorDynamic,
        max_episode_steps=max_episode_steps
    )

    env = gym.make(
        "smarts.env:hiway-v1",
        scenarios=scenarios,
        agent_interfaces={AGENT_ID: agent_interface},
        headless=headless,
    )
    env = SingleAgent(env)

    for episode in episodes(n=num_episodes):
        agent = KeepLaneAgent()
        observation, _ = env.reset()
        episode.record_scenario(env.unwrapped.scenario_log)

        terminated = False
        while not terminated:
            action = agent.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
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
