import os
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DRL:
    def __init__(self):
        if not os.path.exists('model'):
            os.mkdir('model')
        if not os.path.exists('history'):
            os.mkdir('history')

    def test(self, total_episode, delta_flag=False, bartlett_flag=False):
        print('testing...')

        self.epsilon = -1
        w_T_store = []

        for i in range(total_episode):
            observation = self.env.reset()
            done = False
            reward_store = []

            while not done:

                # STATE
                if hasattr(self, "process_obs"):
                    x = self.process_obs(observation).reshape(1, -1)
                else:
                    x = np.array(observation).reshape(1, -1)

                # ACTION
                if delta_flag:
                    action = self.env.delta_path[i % self.env.num_path, self.env.t] * self.env.num_contract * 100

                elif bartlett_flag:
                    action = self.env.bartlett_delta_path[i % self.env.num_path, self.env.t] * self.env.num_contract * 100

                else:
                    action, _, _ = self.egreedy_action(x)

                # STEP
                observation, reward, done, info = self.env.step(action)
                reward_store.append(reward)

            # TERMINAL WEALTH
            w_T = sum(reward_store)
            w_T_store.append(w_T)

        # ================= FINAL METRICS =================
        W = np.array(w_T_store)
        C = -W  # convert to cost

        mean_cost = np.mean(C)
        std_cost = np.std(C)
        median_cost = np.median(C)

        centered = C - mean_cost
        skew = np.mean(centered**3) / (std_cost**3 + 1e-8)
        kurt = np.mean(centered**4) / (std_cost**4 + 1e-8)

        J_cost = (
            mean_cost
            + self.ra_c * std_cost
            + self.skew_c * skew
            + self.kurt_c * kurt
        )

        print("\n====== FINAL RESULTS ======")
        print(f"Mean cost: {mean_cost:.4f}")
        print(f"Std cost: {std_cost:.4f}")
        print(f"Skew cost: {skew:.4f}")
        print(f"Kurt cost: {kurt:.4f}")
        print(f"J cost: {J_cost:.4f}")
        print(f"Median cost: {median_cost:.4f}")

        return {
    "mean_cost": mean_cost,
    "std_cost": std_cost,
    "median_cost": median_cost,
    "skew": skew,
    "kurt": kurt,
    "J_cost": J_cost,
}

    def plot(self, history):
        pass

    def save_history(self, history, name):
        name = os.path.join('history', name)
        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')