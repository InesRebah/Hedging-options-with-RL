import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from drl_higher_moments import DRL
from envs import TradingEnv
from replay_buffer import PrioritizedReplayBuffer
from schedules import LinearSchedule


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

    def __init__(self, env, skew_c=0.05, kurt_c=0.01):
        super().__init__()
        self.skew_c = skew_c # skewness coefficient
        self.kurt_c = kurt_c # kurtosis coefficient
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
        self.ra_c = 1.5      # risk aversion (std penalty)
        self.eps_moment = 1e-8  # numerical stability

        self.epsilon = 1.0
        self.epsilon_decay = 0.997
        self.epsilon_min = 0.05

        self.batch_size = 128
        buffer_size = 600000

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=0.6)
        self.beta_schedule = LinearSchedule(50001, initial_p=0.4, final_p=1.0) \
            if hasattr(LinearSchedule, "__init__") else LinearSchedule(50001, initialp=0.4, finalp=1.0)
        self.prioritized_replay_eps = 1e-6

        self.t = 0

        self.actor = self.build_actor()
        self.actor_target = self.build_actor()
        self.actor_target.set_weights(self.actor.get_weights())

        #Critics (4 moments: mean, second, third, fourth)
        self.critic_q1 = self.build_critic()  # E[R]
        self.critic_q2 = self.build_critic()  # E[R^2]
        self.critic_q3 = self.build_critic()  # E[R^3]
        self.critic_q4 = self.build_critic()  # E[R^4]

        #Target critics
        self.critic_q1_target = self.build_critic()
        self.critic_q2_target = self.build_critic()
        self.critic_q3_target = self.build_critic()
        self.critic_q4_target = self.build_critic()

        #Initialize target weights
        self.critic_q1_target.set_weights(self.critic_q1.get_weights())
        self.critic_q2_target.set_weights(self.critic_q2.get_weights())
        self.critic_q3_target.set_weights(self.critic_q3.get_weights())
        self.critic_q4_target.set_weights(self.critic_q4.get_weights()) 
        
        # --- Optimizers ---
        self.actor_optimizer = Adam(learning_rate=self.actor_lr)
        self.critic_q1_optimizer = Adam(learning_rate=self.critic_lr)  
        self.critic_q2_optimizer = Adam(learning_rate=self.critic_lr)  
        self.critic_q3_optimizer = Adam(learning_rate=self.critic_lr)  
        self.critic_q4_optimizer = Adam(learning_rate=self.critic_lr)  

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
        return q_ex - self.ra_c * tf.sqrt(variance + 1e-8)

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
            # --- Target critics ---
        q1_next = self.critic_q1_target([next_states, next_actions], training=False)
        q2_next = self.critic_q2_target([next_states, next_actions], training=False)
        q3_next = self.critic_q3_target([next_states, next_actions], training=False)
        q4_next = self.critic_q4_target([next_states, next_actions], training=False)

        # --- Targets (moment expansion) ---
        target_q1 = rewards + (1.0 - dones) * q1_next

        target_q2 = tf.square(rewards) + (1.0 - dones) * (
        2.0 * rewards * q1_next + q2_next
        )

        target_q3 = tf.pow(rewards, 3) + (1.0 - dones) * (
    3.0 * tf.square(rewards) * q1_next
    + 3.0 * rewards * q2_next
    + q3_next)

        target_q4 = tf.pow(rewards, 4) + (1.0 - dones) * (
    4.0 * tf.pow(rewards, 3) * q1_next
    + 6.0 * tf.square(rewards) * q2_next
    + 4.0 * rewards * q3_next
    + q4_next)

        # --- Critic Q1 ---
        with tf.GradientTape() as tape1:
            q1_pred = self.critic_q1([states, actions], training=True)
            loss_q1 = tf.reduce_mean(weights * tf.square(target_q1 - q1_pred))
        grads1 = tape1.gradient(loss_q1, self.critic_q1.trainable_variables)
        self.critic_q1_optimizer.apply_gradients(zip(grads1, self.critic_q1.trainable_variables))

        # --- Critic Q2 ---
        with tf.GradientTape() as tape2:
            q2_pred = self.critic_q2([states, actions], training=True)
            loss_q2 = tf.reduce_mean(weights * tf.square(target_q2 - q2_pred))
        grads2 = tape2.gradient(loss_q2, self.critic_q2.trainable_variables)
        self.critic_q2_optimizer.apply_gradients(zip(grads2, self.critic_q2.trainable_variables))

        # --- Critic Q3 ---
        with tf.GradientTape() as tape3:
            q3_pred = self.critic_q3([states, actions], training=True)
            loss_q3 = tf.reduce_mean(weights * tf.square(target_q3 - q3_pred))
        grads3 = tape3.gradient(loss_q3, self.critic_q3.trainable_variables)
        self.critic_q3_optimizer.apply_gradients(zip(grads3, self.critic_q3.trainable_variables))

        # --- Critic Q4 ---
        with tf.GradientTape() as tape4:
            q4_pred = self.critic_q4([states, actions], training=True)
            loss_q4 = tf.reduce_mean(weights * tf.square(target_q4 - q4_pred))
        grads4 = tape4.gradient(loss_q4, self.critic_q4.trainable_variables)
        self.critic_q4_optimizer.apply_gradients(zip(grads4, self.critic_q4.trainable_variables))

        # --- Actor ---
        with tf.GradientTape() as tape5:
            new_actions = self.actor(states, training=True)

            q1_val = self.critic_q1([states, new_actions], training=False)
            q2_val = self.critic_q2([states, new_actions], training=False)
            q3_val = self.critic_q3([states, new_actions], training=False)
            q4_val = self.critic_q4([states, new_actions], training=False)

            var = tf.maximum(q2_val - tf.square(q1_val), self.eps_moment)
            sigma = tf.sqrt(var)

            mu3 = q3_val - 3.0 * q1_val * q2_val + 2.0 * tf.pow(q1_val, 3)
            skew = mu3 / (tf.pow(sigma, 3) + self.eps_moment)
            skew = tf.clip_by_value(skew, -10.0, 10.0)

            mu4 = q4_val - 4.0 * q1_val * q3_val + 6.0 * tf.square(q1_val) * q2_val - 3.0 * tf.pow(q1_val, 4)
            kurt = mu4 / (tf.square(var) + self.eps_moment)
            kurt = tf.clip_by_value(kurt, -10.0, 10.0)

            J = q1_val - self.ra_c * sigma + self.skew_c * skew - self.kurt_c * kurt
            actor_loss = -tf.reduce_mean(J)

        grads5 = tape5.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads5, self.actor.trainable_variables))

        td_error = (tf.abs(q1_pred - target_q1)
    + tf.abs(q2_pred - target_q2)
    + tf.abs(q3_pred - target_q3)
    + tf.abs(q4_pred - target_q4))

        return loss_q1, loss_q2, loss_q3, loss_q4, td_error

    # ------------------------------------------------------------------
    # Target updates
    # ------------------------------------------------------------------
    def update_target_networks(self):
        pairs = [
            (self.actor_target, self.actor),
            (self.critic_q1_target, self.critic_q1),
            (self.critic_q2_target, self.critic_q2),
            (self.critic_q3_target, self.critic_q3),
            (self.critic_q4_target, self.critic_q4),
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
            "loss_q1": [],
            "loss_q2": [],
            "loss_q3": [],
            "loss_q4": []}

        os.makedirs("model", exist_ok=True)

        for ep in range(episodes):
            reset_result = self.env.reset()
            obs = self.process_obs(reset_result)

            done = False
            self.t = ep

            rewards_collected = []
            loss_q1 = np.nan
            loss_q2 = np.nan
            loss_q3 = np.nan
            loss_q4 = np.nan

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

                    loss_q1_t, loss_q2_t, loss_q3_t, loss_q4_t, td_err = self.train_step(
                        states, actions, rewards, next_states, dones, weights
                    )

                    loss_q1 = float(loss_q1_t.numpy())
                    loss_q2 = float(loss_q2_t.numpy())
                    loss_q3 = float(loss_q3_t.numpy())
                    loss_q4 = float(loss_q4_t.numpy())

                    new_priorities = td_err.numpy().reshape(-1) + self.prioritized_replay_eps
                    self.replay_buffer.update_priorities(idxes, new_priorities)

                    self.update_target_networks()

            self.update_epsilon()

            save_dir = f"model/higher_moments/{savetag}" if savetag else "model/higher_moments/default"
            os.makedirs(save_dir, exist_ok=True)
            if ep % 1000 == 0 and ep > 0:
                total_reward = float(np.sum(rewards_collected))
                history["episode"].append(ep)
                history["episode_reward"].append(total_reward)
                history["loss_q1"].append(loss_q1)
                history["loss_q2"].append(loss_q2)
                history["loss_q3"].append(loss_q3)
                history["loss_q4"].append(loss_q4)

                print(
                    f"Episode {ep} | Reward: {total_reward:.4f} | "
                    f"LossQ1: {loss_q1:.6f} | LossQ2: {loss_q2:.6f} | "
                    f"LossQ3: {loss_q3:.6f} | LossQ4: {loss_q4:.6f} | "
                    f"Epsilon: {self.epsilon:.4f}"
                )
    
                tag = f"{savetag}" if savetag else ""
                ckpt = f"{ep // 1000}"
                self.actor.save_weights(f"{save_dir}/ddpg_actor{tag}_{ckpt}.weights.h5")
                self.critic_q1.save_weights(f"{save_dir}/ddpg_critic_q1_{tag}_{ckpt}.weights.h5")
                self.critic_q2.save_weights(f"{save_dir}/ddpg_critic_q2_{tag}_{ckpt}.weights.h5")
                self.critic_q3.save_weights(f"{save_dir}/ddpg_critic_q3_{tag}_{ckpt}.weights.h5")
                self.critic_q4.save_weights(f"{save_dir}/ddpg_critic_q4_{tag}_{ckpt}.weights.h5")

        tag = f"{savetag}" if savetag else ""
        self.actor.save_weights(f"{save_dir}/ddpg_actor_{tag}.weights.h5")
        self.critic_q1.save_weights(f"{save_dir}/ddpg_critic_q1_{tag}.weights.h5")
        self.critic_q2.save_weights(f"{save_dir}/ddpg_critic_q2_{tag}.weights.h5")
        self.critic_q3.save_weights(f"{save_dir}/ddpg_critic_q3_{tag}.weights.h5")
        self.critic_q4.save_weights(f"{save_dir}/ddpg_critic_q4_{tag}.weights.h5")

        return history

    def load(self, tag=""):

        load_dir = f"model/higher_moments/{tag}" if tag else "model/higher_moments/default"
        suffix = f"{tag}" if tag else ""
        self.actor.load_weights(f"{load_dir}/ddpg_actor_{suffix}.weights.h5")
        self.actor_target.load_weights(f"{load_dir}/ddpg_actor_{suffix}.weights.h5")
        self.critic_q1.load_weights(f"{load_dir}/ddpg_critic_q1_{suffix}.weights.h5")
        self.critic_q2.load_weights(f"{load_dir}/ddpg_critic_q2_{suffix}.weights.h5")
        self.critic_q3.load_weights(f"{load_dir}/ddpg_critic_q3_{suffix}.weights.h5")
        self.critic_q4.load_weights(f"{load_dir}/ddpg_critic_q4_{suffix}.weights.h5")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    SEED = args.seed
    
    BASELINE_SABR = {"mu": 0.05, "vol": 0.20, "volvol": 0.60, "beta": 1.0, "rho": -0.4}

    FREQS = {"daily": 1, "weekly": 5}

    TRAIN_EPISODES = 20001
    INIT_TTM = 60
    SPREAD = 0.01
    NUM_CONTRACT = 1

    for freq_name, freq_val in FREQS.items():
        print(f"\n{'=' * 60}\nTRAINING: {freq_name}\n{'=' * 60}")

        env = TradingEnv(
            continuous_action_flag=True,
            sabr_flag=True,
            dg_random_seed=SEED,
            init_ttm=INIT_TTM,
            trade_freq=freq_val,
            spread=SPREAD,
            num_contract=NUM_CONTRACT,
            num_sim=20002,
            model_params=BASELINE_SABR,
        )

        ddpg = DDPG(env, skew_c=0.05, kurt_c=0.01)
        hist = ddpg.train(TRAIN_EPISODES, savetag=f"higher_moments_cpu_run_{freq_name}")

    print("\nAll training runs completed.")