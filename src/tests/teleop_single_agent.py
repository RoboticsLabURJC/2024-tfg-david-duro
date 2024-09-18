import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Final

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

import gymnasium as gym

SMARTS_REPO_PATH = Path(__file__).parents[1].absolute()
sys.path.insert(0, str(SMARTS_REPO_PATH))
from examples.tools.argument_parser import minimal_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from smarts.env.gymnasium.wrappers.single_agent import SingleAgent
from smarts.sstudio.scenario_construction import build_scenarios

AGENT_ID: Final[str] = "Agent"

class KeepLaneAgent(Agent):
    def get_user_input(self):
        print("Select option:")
        print("0: Accelerate")
        print("1: Slow down")
        print("2: Left")
        print("3: Right")
        
        choice = input("Option number: ")
        return choice
    
    def act(self, obs, **kwargs):
        action = self.get_user_input()
        
        # Map the user input to the respective action number
        if action == '1':
            return 1
        elif action == '2':
            return 2
        elif action == '3':
            return 3
        elif action == '0':
            return 0
        else:
            print("Invalid option. Using default option.")
            return 0

def main(scenarios, headless, num_episodes, max_episode_steps=None):
    agent_interface = AgentInterface.from_type(
        AgentType.Laner, max_episode_steps=max_episode_steps
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

