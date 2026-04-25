import os
os.environ["TF_DIRECTML_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from drl import DRL
from envs import TradingEnv
from replay_buffer import PrioritizedReplayBuffer
from schedules import LinearSchedule


gpus = tf.config.list_physical_devices("GPU")
print("Num GPUs Available:", len(gpus))

if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], "GPU")
        print("Using GPU:", gpus[0])
    except Exception as e:
        print("Could not set visible GPU:", e)

for gpu in tf.config.get_visible_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


class DDPG(DRL):
    def __init__(self, env):
        super().__init__()
        self.init(env)

    def init(self, env):
        self.env = env

        self.action_dim = int(np.prod(self.env.action_space.shape))
        initial_obs = self.process_obs(self.env.reset())
        self.state_dim = initial_obs.shape[0]

        self.upper_bound = float(np.asarray(self.env.action_space.high).reshape(-1)[0])
        self.lower_bound = float(np.asarray(self.env.action_space.low).reshape(-1)[0])

        self.TAU = 1e-5
        self.actor_lr = 1e-4
        self.critic_lr = 1e-4

        self.rac = 1.5
        self.ra_c = self.rac

        self.epsilon = 1.0
        self.epsilon_decay = 0.9997
        self.epsilon_min = 0.05

        # self.epsilon = 1.0
        # self.epsilon_decay = 0.99994
        # self.epsilon_min = 0.1

        self.batch_size = 128
        buffer_size = 600000

        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=0.6)
        self.beta_schedule = LinearSchedule(
            schedule_timesteps=50001,
            initial_p=0.4,
            final_p=1.0,
        )
        self.prioritized_replay_eps = 1e-6
        self.t = 0

        self.actor = self.build_actor()
        self.actor_target = self.build_actor()
        self.actor_target.set_weights(self.actor.get_weights())

        self.critic_q_ex = self.build_critic()
        self.critic_q_ex2 = self.build_critic()

        self.critic_q_ex_target = self.build_critic()
        self.critic_q_ex2_target = self.build_critic()

        self.critic_q_ex_target.set_weights(self.critic_q_ex.get_weights())
        self.critic_q_ex2_target.set_weights(self.critic_q_ex2.get_weights())

        self.actor_optimizer = Adam(learning_rate=self.actor_lr)
        self.critic_ex_optimizer = Adam(learning_rate=self.critic_lr)
        self.critic_ex2_optimizer = Adam(learning_rate=self.critic_lr)

    def process_obs(self, obs):
        if isinstance(obs, tuple) and len(obs) == 2 and isinstance(obs[1], dict):
            obs = obs[0]

        flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        return flat

    def process_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.size != self.action_dim:
            raise ValueError(
                f"Action shape mismatch: got {action.shape}, expected ({self.action_dim},)"
            )

        low = np.asarray(self.env.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(self.env.action_space.high, dtype=np.float32).reshape(-1)

        return np.clip(action, low, high).astype(np.float32)

    def build_actor(self):
        inputs = Input(shape=(self.state_dim,))
        x = Dense(32, activation="relu")(inputs)
        x = Dense(64, activation="relu")(x)
        x = Dense(self.action_dim, activation="sigmoid")(x)

        low = tf.constant(np.asarray(self.env.action_space.low, dtype=np.float32).reshape(-1))
        high = tf.constant(np.asarray(self.env.action_space.high, dtype=np.float32).reshape(-1))

        outputs = Lambda(lambda z: low + z * (high - low))(x)

        return Model(inputs, outputs)

    def build_critic(self):
        state_input = Input(shape=(self.state_dim,))
        action_input = Input(shape=(self.action_dim,))

        x = Concatenate()([state_input, action_input])
        x = Dense(32, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        output = Dense(1, activation="linear")(x)

        return Model([state_input, action_input], output)

    def risk_adjusted_q(self, q_ex, q_ex2):
        variance = tf.maximum(q_ex2 - tf.square(q_ex), 0.0)
        return q_ex - self.rac * tf.sqrt(variance + 1e-8)

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
        self.replay_buffer.add(
            np.asarray(state, dtype=np.float32).reshape(-1),
            self.process_action(action),
            np.float32(reward),
            np.asarray(next_state, dtype=np.float32).reshape(-1),
            np.float32(done),
        )

    def sample_batch(self):
        beta = self.beta_schedule.value(self.t)

        states, actions, rewards, next_states, dones, weights, idxes = self.replay_buffer.sample(
            self.batch_size,
            beta=beta,
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
        next_actions = self.actor_target(next_states, training=False)

        q_ex_next = self.critic_q_ex_target([next_states, next_actions], training=False)
        q_ex2_next = self.critic_q_ex2_target([next_states, next_actions], training=False)

        target_q_ex = rewards + (1.0 - dones) * q_ex_next
        target_q_ex2 = tf.square(rewards) + (1.0 - dones) * (
            2.0 * rewards * q_ex_next + q_ex2_next
        )

        target_q_ex = tf.stop_gradient(target_q_ex)
        target_q_ex2 = tf.stop_gradient(target_q_ex2)

        with tf.GradientTape() as tape1:
            q_ex_pred = self.critic_q_ex([states, actions], training=True)
            loss_ex = tf.reduce_mean(weights * tf.square(target_q_ex - q_ex_pred))

        grads1 = tape1.gradient(loss_ex, self.critic_q_ex.trainable_variables)
        self.critic_ex_optimizer.apply_gradients(
            zip(grads1, self.critic_q_ex.trainable_variables)
        )

        with tf.GradientTape() as tape2:
            q_ex2_pred = self.critic_q_ex2([states, actions], training=True)
            loss_ex2 = tf.reduce_mean(weights * tf.square(target_q_ex2 - q_ex2_pred))

        grads2 = tape2.gradient(loss_ex2, self.critic_q_ex2.trainable_variables)
        self.critic_ex2_optimizer.apply_gradients(
            zip(grads2, self.critic_q_ex2.trainable_variables)
        )

        with tf.GradientTape() as tape3:
            new_actions = self.actor(states, training=True)
            q_ex_val = self.critic_q_ex([states, new_actions], training=False)
            q_ex2_val = self.critic_q_ex2([states, new_actions], training=False)
            q_risk = self.risk_adjusted_q(q_ex_val, q_ex2_val)
            actor_loss = -tf.reduce_mean(q_risk)

        grads3 = tape3.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(grads3, self.actor.trainable_variables)
        )

        td_error = tf.abs(q_ex2_pred - target_q_ex2)

        return loss_ex, loss_ex2, td_error

    def update_target_networks(self):
        pairs = [
            (self.actor_target, self.actor),
            (self.critic_q_ex_target, self.critic_q_ex),
            (self.critic_q_ex2_target, self.critic_q_ex2),
        ]

        for target_model, source_model in pairs:
            for target_weight, source_weight in zip(
                target_model.trainable_variables,
                source_model.trainable_variables,
            ):
                target_weight.assign(
                    self.TAU * source_weight + (1.0 - self.TAU) * target_weight
                )

    def train(self, episodes, savetag=""):
        history = {
            "episode": [],
            "episode_reward": [],
            "loss_ex": [],
            "loss_ex2": [],
        }

        os.makedirs("model", exist_ok=True)

        for ep in range(episodes):
            obs = self.process_obs(self.env.reset())
            done = False
            self.t = ep

            rewards_collected = []
            loss_ex = np.nan
            loss_ex2 = np.nan

            while not done:
                state = obs.reshape(1, -1).astype(np.float32)
                action, _, _ = self.egreedy_action(state)

                next_obs_raw, reward, done, info = self.env.step(action)
                next_obs = self.process_obs(next_obs_raw)

                self.remember(obs, action, reward, next_obs, done)

                rewards_collected.append(float(reward))
                obs = next_obs

                if len(self.replay_buffer) > self.batch_size:
                    states, actions, rewards, next_states, dones, weights, idxes = self.sample_batch()

                    loss_ex_t, loss_ex2_t, td_err = self.train_step(
                        states,
                        actions,
                        rewards,
                        next_states,
                        dones,
                        weights,
                    )

                    loss_ex = float(loss_ex_t.numpy())
                    loss_ex2 = float(loss_ex2_t.numpy())

                    new_priorities = td_err.numpy().reshape(-1) + self.prioritized_replay_eps
                    self.replay_buffer.update_priorities(idxes, new_priorities)

                    self.update_target_networks()

            self.update_epsilon()

            if ep % 500 == 0 and ep > 0:           #if ep % 1000 == 0 and ep > 0:
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

                ckpt = f"{ep // 500}"
                self.actor.save_weights(f"model/ddpg_actor{savetag}{ckpt}.weights.h5")
                self.critic_q_ex.save_weights(f"model/ddpg_critic_q_ex{savetag}{ckpt}.weights.h5")
                self.critic_q_ex2.save_weights(f"model/ddpg_critic_q_ex2{savetag}{ckpt}.weights.h5")

        self.actor.save_weights(f"model/ddpg_actor{savetag}.weights.h5")
        self.critic_q_ex.save_weights(f"model/ddpg_critic_q_ex{savetag}.weights.h5")
        self.critic_q_ex2.save_weights(f"model/ddpg_critic_q_ex2{savetag}.weights.h5")

        return history

    def load(self, tag=""):
        self.actor.load_weights(f"model/ddpg_actor{tag}.weights.h5")
        self.actor_target.load_weights(f"model/ddpg_actor{tag}.weights.h5")

        self.critic_q_ex.load_weights(f"model/ddpg_critic_q_ex{tag}.weights.h5")
        self.critic_q_ex_target.load_weights(f"model/ddpg_critic_q_ex{tag}.weights.h5")

        self.critic_q_ex2.load_weights(f"model/ddpg_critic_q_ex2{tag}.weights.h5")
        self.critic_q_ex2_target.load_weights(f"model/ddpg_critic_q_ex2{tag}.weights.h5")


if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)

    FX_PARAMS = {
        "mu": 0.02,
        "vol": 0.10,
        "S": 100,
        "K": 100,
        "rd": 0.04,
        "rf": 0.02,
    }
    
    # COMMODITY_PARAMS = {
    #     "kappa": 2.0,
    #     "theta": np.log(100),
    #     "sigma": 0.30,
    #     "S": 100,
    #     "K": 100,
    #     "r": 0.04,
    #     "q": 0.03,
    #     "sigma_bs": 0.30,
    # }
    # BASELINE_SABR = {
    #     "mu": 0.05,
    #     "vol": 0.20,
    #     "volvol": 0.60,
    #     "beta": 1.0,
    #     "rho": -0.4,
    # }

    FREQS = {"daily": 1}

    TRAIN_EPISODES = 30001
    INIT_TTM = 20
    SPREAD = 0.01
    NUM_CONTRACT = 1
    NUM_SIM_TRAIN = 25000

    for freq_name, freq_val in FREQS.items():
        print(f"\n{'=' * 60}")
        print(f"TRAINING FX MODEL: {freq_name}")
        print(f"{'=' * 60}")

        env = TradingEnv(
            continuous_action_flag=True,
            asset_class="fx",
            sabr_flag=False,
            dg_random_seed=1,
            init_ttm=INIT_TTM,
            trade_freq=freq_val,
            spread=SPREAD,
            num_contract=NUM_CONTRACT,
            num_sim=NUM_SIM_TRAIN,
            model_params=FX_PARAMS,
            domain_randomization=False,
        )

        ddpg = DDPG(env)
        history = ddpg.train(TRAIN_EPISODES, savetag=f"fx_{freq_name}")
        ddpg.save_history(history, f"ddpg_fx_{freq_name}.csv")

        # env = TradingEnv(
        #     continuous_action_flag=True,
        #     asset_class="commodity",
        #     sabr_flag=False,
        #     dg_random_seed=1,
        #     init_ttm=INIT_TTM,
        #     trade_freq=freq_val,
        #     spread=SPREAD,
        #     num_contract=NUM_CONTRACT,
        #     num_sim=NUM_SIM_TRAIN,
        #     model_params=COMMODITY_PARAMS,
        #     domain_randomization=False,
        # )

        # ddpg = DDPG(env)
        # history = ddpg.train(TRAIN_EPISODES, savetag=f"commodity_{freq_name}")
        # ddpg.save_history(history, f"ddpg_commodity_{freq_name}.csv")

        # To train the original SABR baseline later, uncomment:
        #
        # sabr_env = TradingEnv(
        #     continuous_action_flag=True,
        #     asset_class="equity",
        #     sabr_flag=True,
        #     dg_random_seed=1,
        #     init_ttm=INIT_TTM,
        #     trade_freq=freq_val,
        #     spread=SPREAD,
        #     num_contract=NUM_CONTRACT,
        #     num_sim=NUM_SIM_TRAIN,
        #     model_params=BASELINE_SABR,
        #     domain_randomization=False,
        # )
        #
        # ddpg_sabr = DDPG(sabr_env)
        # history_sabr = ddpg_sabr.train(TRAIN_EPISODES, savetag=f"baseline_{freq_name}")
        # ddpg_sabr.save_history(history_sabr, f"ddpg_baseline_{freq_name}.csv")

    print("\nAll training runs completed.")