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


gpus = tf.config.list_physical_devices("GPU")
print("Num GPUs Available:", len(gpus))
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


class DDPG_SMSE(DRL):
    """
    DDPG adapted to terminal SMSE objective.

    Env rewards are used only to reconstruct terminal wealth:
        W_T = sum(accounting P&L rewards)

    Training reward is:
        r_t = 0 for t < T
        r_T = - max(-W_T, 0)^2
    """

    def __init__(self, env):
        super().__init__()
        self.init(env)

    def init(self, env):
        self.env = env

        self.action_dim = int(np.prod(self.env.action_space.shape)) if hasattr(self.env.action_space, "shape") else 1

        reset_result = self.env.reset()
        initial_obs = self.process_obs(reset_result)
        self.state_dim = initial_obs.shape[0]

        if hasattr(self.env, "num_state"):
            self.env.num_state = self.state_dim
        if hasattr(self.env, "numstate"):
            self.env.numstate = self.state_dim

        self.upper_bound = float(np.asarray(self.env.action_space.high).reshape(-1)[0])
        self.lower_bound = float(np.asarray(self.env.action_space.low).reshape(-1)[0])

        self.TAU = 1e-5
        self.actor_lr = 1e-4
        self.critic_lr = 1e-4

        self.epsilon = 1.0
        self.epsilon_decay = 0.997
        self.epsilon_min = 0.05

        self.batch_size = 128
        buffer_size = 600000

        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=0.6)
        self.beta_schedule = LinearSchedule(50001, initial_p=0.4, final_p=1.0)
        self.prioritized_replay_eps = 1e-6
        self.t = 0

        self.actor = self.build_actor()
        self.actor_target = self.build_actor()
        self.actor_target.set_weights(self.actor.get_weights())

        self.critic_q = self.build_critic()
        self.critic_q_target = self.build_critic()
        self.critic_q_target.set_weights(self.critic_q.get_weights())

        self.actor_optimizer = Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = Adam(learning_rate=self.critic_lr)

    def process_obs(self, obs):
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

        return np.asarray(flatten(obs), dtype=np.float32)

    def process_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.size == 1 and self.action_dim > 1:
            action = np.repeat(action, self.action_dim)

        if action.size != self.action_dim:
            raise ValueError(f"Action shape mismatch: got {action.shape}, expected ({self.action_dim},)")

        low = np.asarray(self.env.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(self.env.action_space.high, dtype=np.float32).reshape(-1)

        return np.clip(action, low, high).astype(np.float32)

    def build_actor(self):
        inputs = Input(shape=(self.state_dim,))
        x = BatchNormalization()(inputs)
        x = Dense(32, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(64, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(self.action_dim, activation="sigmoid")(x)

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

    def egreedy_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.actor(state, training=False).numpy()[0]

        return self.process_action(action), None, None

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

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

        return (
            tf.convert_to_tensor(states, dtype=tf.float32),
            tf.convert_to_tensor(actions, dtype=tf.float32),
            tf.convert_to_tensor(rewards.reshape(-1, 1), dtype=tf.float32),
            tf.convert_to_tensor(next_states, dtype=tf.float32),
            tf.convert_to_tensor(dones.reshape(-1, 1), dtype=tf.float32),
            tf.convert_to_tensor(weights.reshape(-1, 1), dtype=tf.float32),
            idxes,
        )

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones, weights):

        # --- Target ---
        next_actions = self.actor_target(next_states, training=False)
        q_next = self.critic_q_target([next_states, next_actions], training=False)
        target_q = rewards + (1.0 - dones) * q_next
        tf.debugging.check_numerics(target_q, "target_q NaN")

        # --- Critic ---
        with tf.GradientTape() as tape:
            q_pred = self.critic_q([states, actions], training=True)
            critic_loss = tf.reduce_mean(weights * tf.square(target_q - q_pred))
        critic_grads = tape.gradient(critic_loss, self.critic_q.trainable_variables)
        critic_grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in critic_grads]
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_q.trainable_variables))

        # --- Actor ---
        with tf.GradientTape() as tape:
            new_actions = self.actor(states, training=True)
            q_val = self.critic_q([states, new_actions], training=False)
            actor_loss = -tf.reduce_mean(q_val)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        actor_grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in actor_grads]
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        td_error = tf.abs(target_q - q_pred)
        return critic_loss, actor_loss, td_error

    def update_target_networks(self):
        pairs = [
            (self.actor_target, self.actor),
            (self.critic_q_target, self.critic_q),
        ]

        for target_model, source_model in pairs:
            for tw, sw in zip(target_model.trainable_variables, source_model.trainable_variables):
                tw.assign(self.TAU * sw + (1.0 - self.TAU) * tw)

    def smse_terminal_reward(self, terminal_wealth):
        terminal_loss = max(-float(terminal_wealth), 0.0)
        reward = -terminal_loss ** 2
        return np.clip(reward, -1000.0, 0.0)

    def train(self, episodes, savetag=""):
        
        history = {
            "episode": [],
            "terminal_wealth": [],
            "smse_reward": [],
            "critic_loss": [],
            "actor_loss": [],
            "mean100_wealth": [],
        }

        os.makedirs("model/smse", exist_ok=True)
        wealth_window = []
        for ep in range(episodes):
            reset_result = self.env.reset()
            obs = self.process_obs(reset_result)

            done = False
            self.t = ep

            episode_transitions = []
            env_rewards = []

            critic_loss = np.nan
            actor_loss = np.nan

            while not done:
                state = obs.reshape(1, -1).astype(np.float32)
                action, _, _ = self.egreedy_action(state)

                step_result = self.env.step(action)

                if len(step_result) == 5:
                    next_obs_raw, env_reward, terminated, truncated, info = step_result
                    done = bool(terminated or truncated)
                else:
                    next_obs_raw, env_reward, done, info = step_result

                next_obs = self.process_obs(next_obs_raw)

                episode_transitions.append((obs, action, next_obs, done))
                env_rewards.append(float(env_reward))

                obs = next_obs

            terminal_wealth = float(np.sum(env_rewards))
            final_reward = self.smse_terminal_reward(terminal_wealth)
            wealth_window.append(terminal_wealth)
            if len(wealth_window) > 100:
                wealth_window.pop(0)
            mean100_wealth = float(np.mean(wealth_window))

            for k, (s, a, ns, d) in enumerate(episode_transitions):
                reward = final_reward if k == len(episode_transitions) - 1 else 0.0
                self.remember(s, a, reward, ns, d)

            if len(self.replay_buffer) > self.batch_size:
                for _ in range(len(episode_transitions)):
                    states, actions, rewards, next_states, dones, weights, idxes = self.sample_batch()

                    critic_loss_t, actor_loss_t, td_err = self.train_step(
                        states, actions, rewards, next_states, dones, weights
                    )

                    critic_loss = float(critic_loss_t.numpy())
                    actor_loss = float(actor_loss_t.numpy())

                    new_priorities = td_err.numpy().reshape(-1) + self.prioritized_replay_eps
                    self.replay_buffer.update_priorities(idxes, new_priorities)

                    self.update_target_networks()

            self.update_epsilon()

            if ep % 1000 == 0 and ep > 0:
                
                history["episode"].append(ep)
                history["terminal_wealth"].append(terminal_wealth)
                history["smse_reward"].append(final_reward)
                history["critic_loss"].append(critic_loss)
                history["actor_loss"].append(actor_loss)
                history.setdefault("mean100_wealth", []).append(mean100_wealth)


                print(f"Episode {ep} | W_T: {terminal_wealth:.4f} | "
    f"Mean100 W_T: {mean100_wealth:.4f} | "
    f"SMSE reward: {final_reward:.4f} | "
    f"CriticLoss: {critic_loss:.6f} | "
    f"ActorLoss: {actor_loss:.6f} | "
    f"Epsilon: {self.epsilon:.4f}")
                
                tag = f"{savetag}" if savetag else ""
                ckpt = f"{ep // 1000}"
                self.actor.save_weights(f"model/smse/ddpg_smse_actor{tag}{ckpt}.weights.h5")
                self.critic_q.save_weights(f"model/smse/ddpg_smse_critic{tag}{ckpt}.weights.h5")

        tag = f"{savetag}" if savetag else ""
        self.actor.save_weights(f"model/smse/ddpg_smse_actor{tag}.weights.h5")
        self.critic_q.save_weights(f"model/smse/ddpg_smse_critic{tag}.weights.h5")



        return history

    def load(self, tag=""):
        suffix = f"{tag}" if tag else ""
        self.actor.load_weights(f"model/smse/ddpg_smse_actor{suffix}.weights.h5")
        self.actor_target.load_weights(f"model/smse/ddpg_smse_actor{suffix}.weights.h5")
        self.critic_q.load_weights(f"model/smse/ddpg_smse_critic{suffix}.weights.h5")
        self.critic_q_target.load_weights(f"model/smse/ddpg_smse_critic{suffix}.weights.h5")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    SEED = args.seed
    
    BASELINE_SABR = {"mu": 0.05, "vol": 0.20, "volvol": 0.60, "beta": 1.0, "rho": -0.4}

    FREQS = {"daily": 1, "weekly": 5}

    TRAIN_EPISODES = 20001
    INIT_TTM = 20
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

        ddpg = DDPG_SMSE(env)
        hist = ddpg.train(TRAIN_EPISODES, savetag=f"_baseline_{freq_name}")

        if hasattr(ddpg, "savehistory"):
            ddpg.savehistory(hist, f"ddpg_smse_{freq_name}_seed{SEED}.csv")

    print("\nAll training runs completed.")


