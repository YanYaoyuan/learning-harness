from __future__ import annotations


class AlfWorld:
    """One deterministic text game; no expert, facts or admissible-action oracle."""

    def __init__(self, game, seed=0, max_steps=50):
        import os
        from pathlib import Path
        os.environ.setdefault("ALFWORLD_DATA", str((Path.cwd() / "data/alfworld").resolve()))
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

        infos = textworld.EnvInfos(won=True, admissible_commands=False)
        env_id = textworld.gym.register_games(
            [str(game)], infos, batch_size=1, asynchronous=False,
            max_episode_steps=max_steps, wrappers=[AlfredDemangler(shuffle=False)],
        )
        self.env = textworld.gym.make(env_id)
        self.env.seed(seed)

    def reset(self):
        observations, _ = self.env.reset()
        return str(observations[0])

    def step(self, action):
        observations, _, dones, infos = self.env.step([action])
        return str(observations[0]), bool(infos["won"][0]), bool(dones[0])

    def close(self):
        self.env.close()

