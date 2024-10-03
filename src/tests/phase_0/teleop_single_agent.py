import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Final

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

import gymnasium as gym
import numpy as np

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

class KeepLaneAgent(Agent):
    def __init__(self):
        self.state = ActuatorDynamicControllerState()

    def get_user_input(self):
        print("Select option:")
        print("0: Accelerate")
        print("1: Slow down")
        print("2: Turn left")
        print("3: Turn right")
        
        choice = input("Option number: ")
        return choice

    def act(self, obs, **kwargs):
        throttle, brake, steering_rate = 0.0, 0.0, 0.0  # Default action

        action = self.get_user_input()

        if action == '0':  # Accelerate
            throttle = 0.8
        elif action == '1':  # Slow down
            brake = 0.8
        elif action == '2':  # Turn left
            steering_rate = -0.5
        elif action == '3':  # Turn right
            steering_rate = 0.5
        else:
            print("Invalid option. Defaulting to no action.")

        return throttle, brake, steering_rate

def main(scenarios, headless, num_episodes, max_episode_steps=None):
    # Configuración de la interfaz del agente para usar el espacio de acción ActuatorDynamic
    agent_interface = AgentInterface(
        action=ActionSpaceType.ActuatorDynamic,  # Aquí se define el espacio de acción
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
