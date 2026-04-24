import os

from ddpg_per import DDPG
from envs import TradingEnv

if __name__ == "__main__":

    BASELINE_SABR = {
                        "mu": 0.05,
                        "vol": 0.2,
                        "volvol": 0.6,
                        "beta": 1.0,
                        "rho": -0.4,
                        }
    HIGH_VOL_SABR = {
                        "mu": 0.05,
                        "vol": 0.3,      # higher than training vol
                        "volvol": 0.6,
                        "beta": 1.0,
                        "rho": -0.4,
                    }
    # disable GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # specify what to test
    delta_action_test = False
    bartlett_action_test = False

    # specify weights file to load
    tag = "49"

    # set init_ttm, spread, and other parameters according to the env that the model is trained

    
    # ------------------------ ORIGINAL PAPER CONFIG ---------------------------------------------------
    # env_test = TradingEnv(continuous_action_flag=True, sabr_flag=True, dg_random_seed=2, spread=0.01, num_contract=1, init_ttm=20, trade_freq=1, num_sim=100001)
    # ---------------------------------------------------------------------------------------------------

    test_env_same = TradingEnv(
        continuous_action_flag=True,
        sabr_flag=True,
        dg_random_seed=2,
        init_ttm=20,
        trade_freq=1,
        spread=0.01,
        num_contract=1,
        num_sim=100001,
        model_params=BASELINE_SABR
    )

    test_env_highvol = TradingEnv(
        continuous_action_flag=True,
        sabr_flag=True,
        dg_random_seed=3,
        init_ttm=20,
        trade_freq=1,
        spread=0.01,
        num_contract=1,
        num_sim=100001,
        model_params=HIGH_VOL_SABR
    )
    
    ddpg_test = DDPG(env_test) # CHANGE THIS 

    print("\n\n***")
    if delta_action_test:
        print("Testing delta actions.")
    else:
        print("Testing agent actions.")
        if tag == "":
            print("tesing the model saved at the end of the training.")
        else:
            print("Testing model saved at " + tag + "K episode.")
        ddpg_test.load(tag=tag)

    ddpg_test.test(100001, delta_flag=delta_action_test, bartlett_flag=bartlett_action_test)

    # for i in range(1, 51):
    #     tag = str(i)
    #     print("****** ", tag)
    #     ddpg_test.load(tag=tag)
    #     ddpg_test.test(3001, delta_flag=delta_action_test)
