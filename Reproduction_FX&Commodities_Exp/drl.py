import os
import numpy as np
import pandas as pd


class DRL:
    def __init__(self):
        os.makedirs("model", exist_ok=True)
        os.makedirs("history", exist_ok=True)

    def test(self, total_episode, delta_flag=False, bartlett_flag=False, verbose_every=1000):
        print("testing...")

        self.epsilon = -1

        w_T_store = []

        for i in range(total_episode):
            observation = self.env.reset()
            done = False

            action_store = []
            reward_store = []

            while not done:
                x = np.asarray(observation, dtype=np.float32).reshape(1, -1)

                if delta_flag:
                    action = (
                        self.env.delta_path[i % self.env.num_path, self.env.t]
                        * self.env.num_contract
                        * 100
                    )

                elif bartlett_flag:
                    if self.env.bartlett_delta_path is None:
                        raise ValueError("Bartlett delta is not available for this asset class.")

                    action = (
                        self.env.bartlett_delta_path[i % self.env.num_path, self.env.t]
                        * self.env.num_contract
                        * 100
                    )

                else:
                    action, _, _ = self.egreedy_action(x)

                action_store.append(action)

                observation, reward, done, info = self.env.step(action)
                reward_store.append(reward)

            w_T = np.sum(reward_store)
            w_T_store.append(w_T)

            if verbose_every is not None and i % verbose_every == 0:
                w_T_mean = np.mean(w_T_store)
                w_T_var = np.var(w_T_store)
                path_row = info["path_row"]

                print(info)
                with np.printoptions(precision=2, suppress=True):
                    print(
                        f"episode: {i} | final wealth: {w_T:.2f}; "
                        f"so far mean wealth: {w_T_mean:.4f}; "
                        f"variance wealth: {w_T_var:.4f}"
                    )
                    print(
                        f"episode: {i} | so far Y(0): "
                        f"{-w_T_mean + self.ra_c * np.sqrt(w_T_var):.4f}"
                    )
                    print(f"episode: {i} | rewards: {np.asarray(reward_store)}")
                    print(f"episode: {i} | action taken: {np.asarray(action_store)}")
                    print(f"episode: {i} | deltas: {self.env.delta_path[path_row] * 100}")
                    print(f"episode: {i} | underlying price: {self.env.path[path_row]}")
                    print(f"episode: {i} | option price: {self.env.option_price_path[path_row] * 100}\n")

        w_T_mean = np.mean(w_T_store)
        w_T_std = np.std(w_T_store)

        mean_cost = -w_T_mean
        sd_cost = w_T_std
        Y0 = mean_cost + self.rac * sd_cost

        print("\nFINAL RESULTS")
        print("Mean cost:", mean_cost)
        print("Std cost:", sd_cost)
        print("Y(0):", Y0)

        return mean_cost, sd_cost, Y0

    def plot(self, history):
        pass

    def save_history(self, history, name):
        path = os.path.join("history", name)
        df = pd.DataFrame.from_dict(history)
        df.to_csv(path, index=False, encoding="utf-8")