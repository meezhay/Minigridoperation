from minigrid_mdp.env.minigrid_env import MiniGridEnv

DAYS_PER_WEEK = 7


class WeekLongEnv(MiniGridEnv):
    """Same grid and reward; one episode is a continuous week instead of one day."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ep_len = 24 * DAYS_PER_WEEK