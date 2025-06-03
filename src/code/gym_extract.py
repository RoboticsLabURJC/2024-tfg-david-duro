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
