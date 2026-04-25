# Deep Hedging with Reinforcement Learning

## About

This is the companion code for the analysis and extension of the paper *Deep Hedging of Derivatives Using Reinforcement Learning* by Jay Cao, Jacky Chen, John Hull, and Zissis Poulos. The paper is available [here](https://ssrn.com/abstract=3514586) at SSRN.

## Requirement

The code requires gym (0.12.1), tensorflow (1.13.1), and keras (2.3.1). *(Note: Some later extensions, such as the Crypto/Jump-Diffusion models, may utilize PyTorch).*

## Usage
The analysis and discussion of the paper methodology and results are extended over four main branches as reported in their respective folders: 

1. **Reproduction and Trading Costs** 
   Reproduction and robustness tests of the paper results. Additionally, while the transaction costs are assumed constant ($k = 0.01$) in the original paper, results are reproduced here with stochastic trading costs, both independent and correlated with the spot price. 
   - *Train* by running `ddpg_per.py` with the desired model specification. This automatically stores model weights across different checkpoints.
   - *Test* and reproduce results and plots with the code in the notebook `plot_notebook.ipynb`.

2. **Objective Function**
   [Insert Sawsanne's description here if she provides one, e.g., Exploring higher-moment risk penalizations].

3. **Asset classes : FX & Commodities Extension**
   To reproduce the experiments on Foreign Exchange (FX) and Commodities, please refer to the folder: `Reproduction_FX&Commodities_Exp/`. 
   Detailed instructions, including parameter settings and execution steps, are provided in the README file within this folder.

4. **Asset classes : Crypto & Equities (Jump-Diffusion) Extension**
   This extension explicitly exposes the Dual-Critic architecture to extreme market discontinuities by replacing standard continuous diffusions with the **Merton Jump-Diffusion** model. It tests the limits of the RL agent under severe liquidations (Crypto, $\lambda=12$) and biased market crashes (Equity, $\lambda=2$) against an analytically misspecified Black-Scholes delta (using effective volatility).
   - Please refer to `Extension_Crypto_Jump.ipynb`.
   - To avoid lengthy re-training loops, you can bypass the training cells and directly load the pre-trained weights (`.pt` files) available inside the `saved_models_ext/` directory to instantly evaluate the 100,000 out-of-sample paths.

## Credits

* The code structure is adapted from [@eijoac](https://github.com/eijoac)'s github project [Deep Hedging with Reinforcement Learning](https://github.com/tdmdal/rl-hedge-2019).
