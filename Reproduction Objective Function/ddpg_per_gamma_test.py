import os
import argparse
import tensorflow as tf

from ddpg_per_gamma import DDPG
from envs import TradingEnv

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="rl")
    args = parser.parse_args()
    mode = args.mode

    BASELINE_SABR = {
        "mu": 0.05,
        "vol": 0.2,
        "volvol": 0.6,
        "beta": 1.0,
        "rho": -0.4,
    }

    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

    delta_action_test = (mode == "delta")
    bartlett_action_test = (mode == "bartlett")

    CKPT = "20"
    BASE_PATH = "model/gamma"

    env_test = TradingEnv(
        continuous_action_flag=True,
        sabr_flag=True,
        dg_random_seed=2,
        init_ttm=20,
        trade_freq=1,
        spread=0.01,
        num_contract=1,
        num_sim=50001,
        model_params=BASELINE_SABR,
        gamma_flag=True
    )

    ddpg_test = DDPG(env_test)

    print("\n\n***")

    if delta_action_test:
        print("Testing delta hedge.")

    elif bartlett_action_test:
        print("Testing Bartlett hedge.")

    else:
        print(f"Testing RL gamma agent (checkpoint {CKPT}k)...")

        ddpg_test.actor.load_weights(
            f"{BASE_PATH}/ddpg_actorgamma_daily{CKPT}.weights.h5"
        )

        ddpg_test.critic_q_ex.load_weights(
            f"{BASE_PATH}/ddpg_critic_q_exgamma_daily{CKPT}.weights.h5"
        )

        ddpg_test.critic_q_ex2.load_weights(
            f"{BASE_PATH}/ddpg_critic_q_ex2gamma_daily{CKPT}.weights.h5"
        )

    ddpg_test.test(
        50001,
        delta_flag=delta_action_test,
        bartlett_flag=bartlett_action_test
    )