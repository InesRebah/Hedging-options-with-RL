import os
import argparse

from ddpg_per_higher_moments import DDPG
from envs import TradingEnv

if __name__ == "__main__":

    # -------- ARGUMENT --------
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="rl")  # rl / delta / bartlett
    args = parser.parse_args()
    mode = args.mode

    # -------- CONFIG --------
    BASELINE_SABR = {
        "mu": 0.05,
        "vol": 0.2,
        "volvol": 0.6,
        "beta": 1.0,
        "rho": -0.4,
    }

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    delta_action_test = (mode == "delta")
    bartlett_action_test = (mode == "bartlett")

    CKPT = "20"
    BASE_PATH = "model/higher_moments/higher_moments_sabr_1m_weekly_gpu_run"

    # -------- ENV --------
    env_test = TradingEnv(
        continuous_action_flag=True,
        sabr_flag=True,
        dg_random_seed=2,
        init_ttm=20,
        trade_freq=5,
        spread=0.01,
        num_contract=1,
        num_sim=50001,
        model_params=BASELINE_SABR
    )

    ddpg_test = DDPG(env_test)

    print("\n\n***")

    if delta_action_test:
        print("Testing delta actions.")

    elif bartlett_action_test:
        print("Testing Bartlett actions.")

    else:
        print(f"Testing RL agent (checkpoint {CKPT}k)...")

        ddpg_test.actor.load_weights(
            f"{BASE_PATH}/ddpg_actorhigher_moments_weekly_{CKPT}.weights.h5"
        )
        ddpg_test.critic_q1.load_weights(
            f"{BASE_PATH}/ddpg_critic_q1_higher_moments_weekly_{CKPT}.weights.h5"
        )
        ddpg_test.critic_q2.load_weights(
            f"{BASE_PATH}/ddpg_critic_q2_higher_moments_weekly_{CKPT}.weights.h5"
        )
        ddpg_test.critic_q3.load_weights(
            f"{BASE_PATH}/ddpg_critic_q3_higher_moments_weekly_{CKPT}.weights.h5"
        )
        ddpg_test.critic_q4.load_weights(
            f"{BASE_PATH}/ddpg_critic_q4_higher_moments_weekly_{CKPT}.weights.h5"
        )

    ddpg_test.test(
        50001,
        delta_flag=delta_action_test,
        bartlett_flag=bartlett_action_test
    )