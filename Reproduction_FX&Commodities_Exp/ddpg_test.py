import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from ddpg_per import DDPG
import numpy as np
from envs import TradingEnv


if __name__ == "__main__":
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

    TEST_EPISODES = 50000
    INIT_TTM = 20
    TRADE_FREQ = 1
    SPREAD = 0.01
    NUM_CONTRACT = 1
    NUM_SIM_TEST = 50000
    tag = "fx_daily"


    
    # test_env = TradingEnv(
    #     continuous_action_flag=True,
    #     asset_class="commodity",
    #     sabr_flag=False,
    #     dg_random_seed=2,
    #     init_ttm=INIT_TTM,
    #     trade_freq=TRADE_FREQ,
    #     spread=SPREAD,
    #     num_contract=NUM_CONTRACT,
    #     num_sim=NUM_SIM_TEST,
    #     model_params=COMMODITY_PARAMS,
    #     domain_randomization=False,
    # )
    test_env = TradingEnv(
        continuous_action_flag=True,
        asset_class="fx",
        sabr_flag=False,
        dg_random_seed=2,
        init_ttm=INIT_TTM,
        trade_freq=TRADE_FREQ,
        spread=SPREAD,
        num_contract=NUM_CONTRACT,
        num_sim=NUM_SIM_TEST,
        model_params=FX_PARAMS,
        domain_randomization=False,
    )
    ddpg_test = DDPG(test_env)

    print("\nLoading RL model:", tag)
    ddpg_test.load(tag=tag)

    print("\n=== RL HEDGING FX ===")

    rl_mean, rl_sd, Y_rl = ddpg_test.test(
        TEST_EPISODES,
        delta_flag=False,
        bartlett_flag=False,
    )

    print("\n=== FX DELTA HEDGING ===")
    delta_mean, delta_sd, Y_delta = ddpg_test.test(
        TEST_EPISODES,
        delta_flag=True,
        bartlett_flag=False,
    )

    improvement_vs_delta = (Y_delta - Y_rl) / Y_delta

    print("\n=== FX RESULTS ===")
    print("RL mean cost:", rl_mean)
    print("RL std cost:", rl_sd)
    print("RL Y(0):", Y_rl)
    print()
    print("FX Delta mean cost:", delta_mean)
    print("FX Delta std cost:", delta_sd)
    print("FX Delta Y(0):", Y_delta)
    print()
    print("Improvement vs FX Delta (%):", 100 * improvement_vs_delta)

    # print("\n=== RL HEDGING COMMODITY ===")
    # rl_mean, rl_sd, Y_rl = ddpg_test.test(
    #     TEST_EPISODES,
    #     delta_flag=False,
    #     bartlett_flag=False,
    # )

    # print("\n=== COMMODITY DELTA HEDGING ===")
    # delta_mean, delta_sd, Y_delta = ddpg_test.test(
    #     TEST_EPISODES,
    #     delta_flag=True,
    #     bartlett_flag=False,
    # )

    # improvement_vs_delta = (Y_delta - Y_rl) / Y_delta

    # print("\n=== COMMODITY RESULTS ===")
    # print("RL mean cost:", rl_mean)
    # print("RL std cost:", rl_sd)
    # print("RL Y(0):", Y_rl)
    # print()
    # print("Commodity Delta mean cost:", delta_mean)
    # print("Commodity Delta std cost:", delta_sd)
    # print("Commodity Delta Y(0):", Y_delta)
    # print()
    # print("Improvement vs Commodity Delta (%):", 100 * improvement_vs_delta)

    
    # To test the original SABR baseline later, uncomment and adapt:
    #
    # test_env_sabr = TradingEnv(
    #     continuous_action_flag=True,
    #     asset_class="equity",
    #     sabr_flag=True,
    #     dg_random_seed=2,
    #     init_ttm=INIT_TTM,
    #     trade_freq=TRADE_FREQ,
    #     spread=SPREAD,
    #     num_contract=NUM_CONTRACT,
    #     num_sim=NUM_SIM_TEST,
    #     model_params=BASELINE_SABR,
    #     domain_randomization=False,
    # )
    #
    # ddpg_test_sabr = DDPG(test_env_sabr)
    # ddpg_test_sabr.load(tag="baseline_daily20")