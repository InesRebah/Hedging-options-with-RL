"""A trading environment"""
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

import numpy as np

from utils import get_sim_path, get_sim_path_sabr


class TradingEnv(gym.Env):
    """
    trading environment
    """

    # trade_freq in unit of day, e.g 2: every 2 day; 0.5 twice a day
    def __init__(
        self,
        cash_flow_flag=0,
        dg_random_seed=1,
        num_sim=500002,
        sabr_flag=False,
        continuous_action_flag=False,
        spread=0,
        init_ttm=20,
        trade_freq=1,
        num_contract=1,
        model_params=None,
        domain_randomization=False,
        random_param_ranges=None, 
        gamma_flag=False,
        lambda_gamma=0.01
    ):

        self.sabr_flag = sabr_flag
        self.dg_random_seed = dg_random_seed
        self.num_sim = num_sim
        self.init_ttm = init_ttm
        self.trade_freq = trade_freq
        self.model_params = model_params or {}
        self.domain_randomization = domain_randomization
        self.random_param_ranges = random_param_ranges or {}
        self.gamma_flag = gamma_flag
        self.lambda_gamma = lambda_gamma

        # spread
        self.spread = spread

        self.num_contract = num_contract
        self.strike_price = 100

        # track the index of simulated path in use
        self.sim_episode = -1

        # track time step within an episode
        self.t = None

        # action space
        if continuous_action_flag:
            self.action_space = spaces.Box(
                low=np.array([0]),
                high=np.array([num_contract * 100]),
                dtype=np.float32
            )
        else:
            self.num_action = num_contract * 100 + 1
            self.action_space = spaces.Discrete(self.num_action)

        self.num_state = 3
        self.state = []

        # step function initialization depending on cash_flow_flag
        if cash_flow_flag == 1:
            self.step = self.step_cash_flow
        elif gamma_flag:
            self.step = self.step_profit_loss_gamma
        else:
            self.step = self.step_profit_loss

        # seed and start
        self.seed()

        # initialize paths only if NOT doing domain randomization
        if not self.domain_randomization:
            self.current_model_params = self.model_params.copy()
            self._generate_paths(self.current_model_params, seed=self.dg_random_seed)
        else:
            self.current_model_params = None
            self.path = None
            self.option_price_path = None
            self.delta_path = None
            self.bartlett_delta_path = None
            self.num_path = None
            self.num_period = None
            self.ttm_array = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _sample_model_params(self):
        params = {}

        for key, values in self.random_param_ranges.items():
            # interval case: (low, high)
            if isinstance(values, tuple) and len(values) == 2:
                low, high = values
                params[key] = self.np_random.uniform(low, high)

            # discrete set case: [v1, v2, v3, ...]
            elif isinstance(values, list):
                params[key] = self.np_random.choice(values)

            else:
                raise ValueError(
                    f"Unsupported random_param_ranges format for key={key}: {values}"
                )

        return params

    def _generate_paths(self, model_params, seed=None):
        if seed is None:
            seed = int(self.np_random.integers(1, 10**9))

        if self.sabr_flag:
            self.path, self.option_price_path, self.delta_path, self.bartlett_delta_path = get_sim_path_sabr(
                M=self.init_ttm,
                freq=self.trade_freq,
                np_seed=seed,
                num_sim=self.num_sim,
                **model_params
            )
        else:
            self.path, self.option_price_path, self.delta_path = get_sim_path(
                M=self.init_ttm,
                freq=self.trade_freq,
                np_seed=seed,
                num_sim=self.num_sim,
                **model_params
            )

        self.num_path = self.path.shape[0]
        self.num_period = self.path.shape[1]
        self.ttm_array = np.arange(self.init_ttm, -self.trade_freq, -self.trade_freq)

    def reset(self):
        if self.domain_randomization:
            sampled_params = self._sample_model_params()
            self.current_model_params = sampled_params
            self._generate_paths(sampled_params)

            # always start from first path in the newly generated block
            self.sim_episode = 0
        else:
            # repeatedly go through available simulated paths
            self.sim_episode = (self.sim_episode + 1) % self.num_path

        self.t = 0

        price = self.path[self.sim_episode, self.t]
        position = 0
        ttm = self.ttm_array[self.t]

        self.state = [price, position, ttm]

        return self.state

    def step_cash_flow(self, action):
        """
        cash flow period reward
        """

        current_price = self.state[0]
        current_position = self.state[1]

        self.t = self.t + 1

        price = self.path[self.sim_episode, self.t]
        position = action
        ttm = self.ttm_array[self.t]

        self.state = [price, position, ttm]

        cash_flow = -(position - current_position) * current_price - np.abs(position - current_position) * current_price * self.spread

        if self.t == self.num_period - 1:
            done = True
            reward = cash_flow + price * position - max(price - self.strike_price, 0) * self.num_contract * 100 - position * price * self.spread
        else:
            done = False
            reward = cash_flow

        info = {"path_row": self.sim_episode, "model_params": self.current_model_params}

        return self.state, float(np.asarray(reward).reshape(-1)[0]), done, info

    def step_profit_loss(self, action):
        """
        profit loss period reward
        """

        current_price = self.state[0]
        current_option_price = self.option_price_path[self.sim_episode, self.t]
        current_position = self.state[1]

        self.t = self.t + 1

        price = self.path[self.sim_episode, self.t]
        option_price = self.option_price_path[self.sim_episode, self.t]
        position = action
        ttm = self.ttm_array[self.t]

        self.state = [price, position, ttm]

        reward = (price - current_price) * position - np.abs(current_position - position) * current_price * self.spread

        if self.t == self.num_period - 1:
            done = True
            reward = reward - (max(price - self.strike_price, 0) - current_option_price) * self.num_contract * 100 - position * price * self.spread
        else:
            done = False
            reward = reward - (option_price - current_option_price) * self.num_contract * 100

        info = {"path_row": self.sim_episode, "model_params": self.current_model_params}

        return self.state, float(np.asarray(reward).reshape(-1)[0]), done, info
    

    def step_profit_loss_gamma(self, action):
        """
        profit & loss reward + gamma-like penalty
        """

        current_price = self.state[0]
        current_option_price = self.option_price_path[self.sim_episode, self.t]
        current_position = self.state[1]

        self.t = self.t + 1

        price = self.path[self.sim_episode, self.t]
        option_price = self.option_price_path[self.sim_episode, self.t]
        position = action
        ttm = self.ttm_array[self.t]

        self.state = [price, position, ttm]

        reward = (price - current_price) * position \
             - np.abs(current_position - position) * current_price * self.spread

        if self.t == self.num_period - 1:
            done = True
            reward = reward - (max(price - self.strike_price, 0) - current_option_price) \
                 * self.num_contract * 100 \
                 - position * price * self.spread
        else:
            done = False
            reward = reward - (option_price - current_option_price) \
                 * self.num_contract * 100

        delta_H = position - current_position
        delta_S = price - current_price

        gamma_like = 0.5 * delta_H * delta_S

        reward -= self.lambda_gamma * (gamma_like ** 2)

        info = {"path_row": self.sim_episode,
        "model_params": self.current_model_params}

        return self.state, float(np.asarray(reward).reshape(-1)[0]), done, info