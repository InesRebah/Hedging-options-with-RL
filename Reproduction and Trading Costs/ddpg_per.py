import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from drl import DRL
from envs import TradingEnv
from replay_buffer import PrioritizedReplayBuffer
from schedules import LinearSchedule
import pandas as pd

gpus = tf.config.list_physical_devices("GPU")
print("Num GPUs Available:", len(gpus))
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


class DDPG(DRL):
    """
    DDPG with prioritized replay, adapted for:
    - TensorFlow 2.x / Keras 3
    - NumPy 2.x
    - Gymnasium/Gym API differences
    """

    def __init__(self, env):
        super().__init__()
        self.init(env)

    def init(self, env):
        self.env = env

        # Robust action/state dimension discovery
        self.action_dim = int(np.prod(self.env.action_space.shape)) if hasattr(self.env.action_space, "shape") else 1

        reset_result = self.env.reset()
        initial_obs = self.process_obs(reset_result)
        self.state_dim = initial_obs.shape[0]

        # Keep compatibility if old code expects env.num_state
        if hasattr(self.env, "num_state"):
            self.env.num_state = self.state_dim
        if hasattr(self.env, "numstate"):
            self.env.numstate = self.state_dim

        self.upper_bound = float(np.asarray(self.env.action_space.high).reshape(-1)[0])
        self.lower_bound = float(np.asarray(self.env.action_space.low).reshape(-1)[0])

        # Hyperparameters
        self.TAU = 1e-5
        self.actor_lr = 1e-4
        self.critic_lr = 1e-4
        self.rac = 1.5

        self.epsilon = 1.0
        self.epsilon_decay = 0.9997    #0.99994 for 50k we adapt to 20k
        self.epsilon_min = 0.1

        self.batch_size = 128
        buffer_size = 600000

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=0.6)
        self.beta_schedule = LinearSchedule(50001, initial_p=0.4, final_p=1.0) \
            if hasattr(LinearSchedule, "__init__") else LinearSchedule(50001, initialp=0.4, finalp=1.0)
        self.prioritized_replay_eps = 1e-6

        self.t = 0

        # Networks
        self.actor = self.build_actor()
        self.actor_target = self.build_actor()
        self.actor_target.set_weights(self.actor.get_weights())

        self.critic_q_ex = self.build_critic()
        self.critic_q_ex2 = self.build_critic()
        self.critic_q_ex_target = self.build_critic()
        self.critic_q_ex2_target = self.build_critic()

        self.critic_q_ex_target.set_weights(self.critic_q_ex.get_weights())
        self.critic_q_ex2_target.set_weights(self.critic_q_ex2.get_weights())

        # Optimizers
        self.actor_optimizer = Adam(learning_rate=self.actor_lr)
        self.critic_ex_optimizer = Adam(learning_rate=self.critic_lr)
        self.critic_ex2_optimizer = Adam(learning_rate=self.critic_lr)

    # ------------------------------------------------------------------
    # Observation processing
    # ------------------------------------------------------------------
    def process_obs(self, obs):
        """
        Convert any observation format into a flat float32 vector.
        Handles:
        - Gymnasium reset() -> (obs, info)
        - lists/tuples/nested arrays/scalars
        """
        if isinstance(obs, tuple) and len(obs) == 2 and isinstance(obs[1], dict):
            obs = obs[0]

        def flatten(x):
            out = []
            if isinstance(x, np.ndarray):
                out.extend(np.asarray(x, dtype=np.float32).reshape(-1).tolist())
            elif isinstance(x, (list, tuple)):
                for item in x:
                    out.extend(flatten(item))
            elif isinstance(x, (int, float, np.integer, np.floating, bool, np.bool_)):
                out.append(float(x))
            else:
                arr = np.asarray(x, dtype=np.float32).reshape(-1)
                out.extend(arr.tolist())
            return out

        flat = flatten(obs)
        return np.asarray(flat, dtype=np.float32)

    def process_action(self, action):
        """
        Force every action into shape (action_dim,) float32.
        This is the key fix for replay buffer shape consistency.
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.size == 1 and self.action_dim > 1:
            action = np.repeat(action, self.action_dim)

        if action.size != self.action_dim:
            raise ValueError(
                f"Action shape mismatch: got {action.shape}, expected ({self.action_dim},)"
            )

        # Clip to env bounds
        low = np.asarray(self.env.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(self.env.action_space.high, dtype=np.float32).reshape(-1)
        action = np.clip(action, low, high)

        return action.astype(np.float32)

    # ------------------------------------------------------------------
    # Networks
    # ------------------------------------------------------------------
    def build_actor(self):
        inputs = Input(shape=(self.state_dim,))
        x = BatchNormalization()(inputs)
        x = Dense(32, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(64, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(self.action_dim, activation="sigmoid")(x)

        # Scale [0,1] -> [low, high]
        low = tf.constant(np.asarray(self.env.action_space.low, dtype=np.float32).reshape(-1))
        high = tf.constant(np.asarray(self.env.action_space.high, dtype=np.float32).reshape(-1))

        output = Lambda(lambda z: low + z * (high - low))(x)
        return Model(inputs, output)

    def build_critic(self):
        state_input = Input(shape=(self.state_dim,))
        action_input = Input(shape=(self.action_dim,))
        x = Concatenate()([state_input, action_input])
        x = BatchNormalization()(x)
        x = Dense(32, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(64, activation="relu")(x)
        x = BatchNormalization()(x)
        output = Dense(1, activation="linear")(x)
        return Model([state_input, action_input], output)

    # ------------------------------------------------------------------
    # Risk-adjusted Q
    # ------------------------------------------------------------------
    def risk_adjusted_q(self, q_ex, q_ex2):
        variance = tf.maximum(q_ex2 - tf.square(q_ex), 0.0)
        return q_ex - self.rac * tf.sqrt(variance + 1e-8)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def egreedy_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.actor(state, training=False).numpy()[0]

        action = self.process_action(action)
        return action, None, None

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    # ------------------------------------------------------------------
    # Replay buffer helpers
    # ------------------------------------------------------------------
    def remember(self, state, action, reward, next_state, done):
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        action = self.process_action(action)
        next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)
        reward = np.float32(reward)
        done = np.float32(done)

        self.replay_buffer.add(state, action, reward, next_state, done)

    def sample_batch(self):
        beta = self.beta_schedule.value(self.t)

        states, actions, rewards, next_states, dones, weights, idxes = self.replay_buffer.sample(
            self.batch_size, beta=beta
        )

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards.reshape(-1, 1), dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones.reshape(-1, 1), dtype=tf.float32)
        weights = tf.convert_to_tensor(weights.reshape(-1, 1), dtype=tf.float32)

        return states, actions, rewards, next_states, dones, weights, idxes

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones, weights):
        next_actions = self.actor_target(next_states, training=False)
        q_ex_next = self.critic_q_ex_target([next_states, next_actions], training=False)
        q_ex2_next = self.critic_q_ex2_target([next_states, next_actions], training=False)

        target_q_ex = rewards + (1.0 - dones) * q_ex_next
        target_q_ex2 = tf.square(rewards) + (1.0 - dones) * (2.0 * rewards * q_ex_next + q_ex2_next)

        with tf.GradientTape() as tape1:
            q_ex_pred = self.critic_q_ex([states, actions], training=True)
            loss_ex = tf.reduce_mean(weights * tf.square(target_q_ex - q_ex_pred))
        grads1 = tape1.gradient(loss_ex, self.critic_q_ex.trainable_variables)
        self.critic_ex_optimizer.apply_gradients(zip(grads1, self.critic_q_ex.trainable_variables))

        with tf.GradientTape() as tape2:
            q_ex2_pred = self.critic_q_ex2([states, actions], training=True)
            loss_ex2 = tf.reduce_mean(weights * tf.square(target_q_ex2 - q_ex2_pred))
        grads2 = tape2.gradient(loss_ex2, self.critic_q_ex2.trainable_variables)
        self.critic_ex2_optimizer.apply_gradients(zip(grads2, self.critic_q_ex2.trainable_variables))

        with tf.GradientTape() as tape3:
            new_actions = self.actor(states, training=True)
            q_ex_val = self.critic_q_ex([states, new_actions], training=False)
            q_ex2_val = self.critic_q_ex2([states, new_actions], training=False)
            q_risk = self.risk_adjusted_q(q_ex_val, q_ex2_val)
            actor_loss = -tf.reduce_mean(q_risk)
        grads3 = tape3.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads3, self.actor.trainable_variables))

        td_error = tf.abs(q_ex2_pred - target_q_ex2)
        return loss_ex, loss_ex2, td_error

    # ------------------------------------------------------------------
    # Target updates
    # ------------------------------------------------------------------
    def update_target_networks(self):
        pairs = [
            (self.actor_target, self.actor),
            (self.critic_q_ex_target, self.critic_q_ex),
            (self.critic_q_ex2_target, self.critic_q_ex2),
        ]

        for target_model, source_model in pairs:
            for tw, sw in zip(target_model.trainable_variables, source_model.trainable_variables):
                tw.assign(self.TAU * sw + (1.0 - self.TAU) * tw)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self, episodes, savetag=""):
        history = {
            "episode": [],
            "episode_reward": [],
            "loss_ex": [],
            "loss_ex2": [],
        }

        os.makedirs("model", exist_ok=True)

        for ep in range(episodes):
            reset_result = self.env.reset()
            obs = self.process_obs(reset_result)

            done = False
            self.t = ep

            rewards_collected = []
            loss_ex = np.nan
            loss_ex2 = np.nan

            while not done:
                state = obs.reshape(1, -1).astype(np.float32)
                action, _, _ = self.egreedy_action(state)

                step_result = self.env.step(action)

                if len(step_result) == 5:
                    next_obs_raw, reward, terminated, truncated, info = step_result
                    done = bool(terminated or truncated)
                else:
                    next_obs_raw, reward, done, info = step_result

                next_obs = self.process_obs(next_obs_raw)

                self.remember(obs, action, reward, next_obs, done)
                rewards_collected.append(float(reward))
                obs = next_obs

                if len(self.replay_buffer) > self.batch_size:
                    states, actions, rewards, next_states, dones, weights, idxes = self.sample_batch()

                    loss_ex_t, loss_ex2_t, td_err = self.train_step(
                        states, actions, rewards, next_states, dones, weights
                    )

                    loss_ex = float(loss_ex_t.numpy())
                    loss_ex2 = float(loss_ex2_t.numpy())

                    new_priorities = td_err.numpy().reshape(-1) + self.prioritized_replay_eps
                    self.replay_buffer.update_priorities(idxes, new_priorities)

                    self.update_target_networks()

            self.update_epsilon()

            if ep % 1000 == 0 and ep > 0:
                total_reward = float(np.sum(rewards_collected))
                history["episode"].append(ep)
                history["episode_reward"].append(total_reward)
                history["loss_ex"].append(loss_ex)
                history["loss_ex2"].append(loss_ex2)

                print(
                    f"Episode {ep} | Reward: {total_reward:.4f} | "
                    f"LossEx: {loss_ex:.6f} | LossEx2: {loss_ex2:.6f} | "
                    f"Epsilon: {self.epsilon:.4f}"
                )

                tag = f"{savetag}" if savetag else ""
                ckpt = f"{ep // 1000}"

                self.actor.save_weights(f"model/ddpg_actor{tag}{ckpt}.weights.h5")
                self.critic_q_ex.save_weights(f"model/ddpg_critic_q_ex{tag}{ckpt}.weights.h5")
                self.critic_q_ex2.save_weights(f"model/ddpg_critic_q_ex2{tag}{ckpt}.weights.h5")

        tag = f"{savetag}" if savetag else ""
        self.actor.save_weights(f"model/ddpg_actor{tag}.weights.h5")
        self.critic_q_ex.save_weights(f"model/ddpg_critic_q_ex{tag}.weights.h5")
        self.critic_q_ex2.save_weights(f"model/ddpg_critic_q_ex2{tag}.weights.h5")

        return history

    def load(self, tag=""):
        suffix = f"{tag}" if tag else ""
        self.actor.load_weights(f"model/ddpg_actor{suffix}.weights.h5")
        self.actor_target.load_weights(f"model/ddpg_actor{suffix}.weights.h5")

        self.critic_q_ex.load_weights(f"model/ddpg_critic_q_ex{suffix}.weights.h5")
        self.critic_q_ex_target.load_weights(f"model/ddpg_critic_q_ex{suffix}.weights.h5")

        self.critic_q_ex2.load_weights(f"model/ddpg_critic_q_ex2{suffix}.weights.h5")
        self.critic_q_ex2_target.load_weights(f"model/ddpg_critic_q_ex2{suffix}.weights.h5")


if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)

    BASELINE_SABR = {"mu": 0.05, "vol": 0.20, "volvol": 0.60, "beta": 1.0, "rho": -0.4}

    RANDOM_SABR_RANGES = {
        "mu": [0.05],
        "vol": [0.15, 0.20, 0.25, 0.30],
        "volvol": [0.4, 0.6, 0.8],
        "beta": [1.0],
        "rho": [-0.4],
    }

    TRAIN_EPISODES = 50001 # or 20000
    INIT_TTM = 20
    SPREAD = 0.01
    NUM_CONTRACT = 1
    freq_name = 'daily'
    freq_val = 1

    print(f"\n{'=' * 60}\nTRAINING: {freq_name}\n{'=' * 60}")
    
    # ----------------------------------------------------------
    # --------------- Fixed param Training ----------------------
    # ----------------------------------------------------------

    baseline_env = TradingEnv(
        continuous_action_flag=True,
        sabr_flag=True,
        dg_random_seed=1,
        init_ttm=INIT_TTM,
        trade_freq=freq_val,
        spread=SPREAD,
        num_contract=NUM_CONTRACT,
        num_sim=50002,
        model_params=BASELINE_SABR,
        domain_randomization=False,
    
        stochastic_tc=False,
        lambda_bar=0.01,
        kappa=1.0,
        xi=0.3,
        lambda_spot_corr=0.0,   
    )

    ddpg = DDPG(baseline_env)
        
    # ----------------------------------------------------------
    # --------------- Domain randomization Training ------------
    # ----------------------------------------------------------

    rand_env = TradingEnv(
        continuous_action_flag=True,
        sabr_flag=True,
        dg_random_seed=10,
        init_ttm=INIT_TTM,
        trade_freq=freq_val,
        spread=SPREAD,
        num_contract=NUM_CONTRACT,
        num_sim=2000,
        domain_randomization=True,
        random_param_ranges=RANDOM_SABR_RANGES,
    
        stochastic_tc=False,
        lambda_bar=0.01,
        kappa=1.0,
        xi=0.3,
        lambda_spot_corr=0.0,   
        )

    ddpg_rand = DDPG(rand_env)

    # ----------------------------------------------------------
    # --------------- Stoch TC independent ---------------------
    # ----------------------------------------------------------

    stoch_env = TradingEnv(
        continuous_action_flag=True,
        sabr_flag=True,
        dg_random_seed=1,
        init_ttm=INIT_TTM,
        trade_freq=1,
        num_contract=NUM_CONTRACT,
        num_sim=20000,
        model_params=BASELINE_SABR,
        domain_randomization=False,

        stochastic_tc=True,
        lambda_bar=0.01,
        kappa=1.0,
        xi=0.3,
        lambda_spot_corr=0.0,   # independent
    )

    ddpg = DDPG(stoch_env)

    hist = ddpg.train(
        TRAIN_EPISODES,
        savetag=f"stochTC_indep_{freq_name}"
    )

    pd.DataFrame(hist).to_csv(
        f"history/ddpg_baseline_{freq_name}_stochTC_indep.csv",
        index=False
    )
    
    # ----------------------------------------------------------
    # --------------- Stoch TC correlated ---------------------
    # ----------------------------------------------------------

    corr_env = TradingEnv(
        continuous_action_flag=True,
        sabr_flag=True,
        dg_random_seed=1,
        init_ttm=INIT_TTM,
        trade_freq=1,
        num_contract=NUM_CONTRACT,
        num_sim=20000,
        model_params=BASELINE_SABR,
        domain_randomization=False,

        stochastic_tc=True,
        lambda_bar=0.01,
        kappa=1.0,
        xi=0.3,
        lambda_spot_corr=-0.7,   # correlated
    )

    ddpg = DDPG(corr_env)

    hist = ddpg.train(
        TRAIN_EPISODES,
        savetag=f"stochTC_corr_{freq_name}"
    )
    pd.DataFrame(hist).to_csv(
        f"history/ddpg_baseline_{freq_name}stochTC_corr.csv",
        index=False
    )
    print("\nAll training runs completed.")