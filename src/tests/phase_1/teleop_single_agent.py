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

from smarts.core.controllers.direct_controller import DirectController

AGENT_ID: Final[str] = "Agent"

class KeepLaneAgent(Agent):
    def __init__(self):
        self.state = DirectController()

    def get_user_input(self):
        print("Select option:")
        print("0: Accelerate")
        print("1: Go back")
        print("2: Turn left")
        print("3: Turn right")
        
        choice = input("Option number: ")
        return choice

    def act(self, obs, **kwargs):
        v, w = 0.0, 0.0  # Default action

        action = self.get_user_input()

        if action == '0':  # Accelerate
            v = 2
        elif action == '1':  # Slow down
            v = -20
        elif action == '2':  # Turn left
            v = 1
            w = -0.5
        elif action == '3':  # Turn right
            v = 1
            w = 0.5
        else:
            print("Invalid option. Defaulting to no action.")

        return v, w

def main(scenarios, headless, num_episodes, max_episode_steps=None):
    agent_interface = AgentInterface(
        action=ActionSpaceType.Direct,  # Aquí se define el espacio de acción
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
