# Deep Hedging with Reinforcement Learning

## About

This is the companion code for the analysis and extension of the paper *Deep Hedging of Derivatives Using Reinforcement Learning* by Jay Cao, Jacky Chen, John Hull, and Zissis Poulos. The paper is available [here](https://ssrn.com/abstract=3514586) at SSRN.

## Requirement

The code requires gym (0.12.1), tensorflow (1.13.1), and keras (2.3.1).

## Usage
The analysis and discussion of the paper methodology and results are extended over three main branches as reported in the folders : 
1. Reproduction and Trading Costs : Reproduction and robustness tests of the paper results. Additionaly, the transaction costs assumed constant k = 0.01 in the paper and results are reproduced with stochastic trading costs booth independant and correlated with the spot price. \\
  Train by running ddpg_per.py with model specification. It automatically stores model weights across diffrent checkpoints. \\
  Test and plots reproduction in notebook plot_notebook.ipynb
3. Objective Function :

4. Asset classes : 

## Credits

* The code structure is adapted from [@xiaochus](https://github.com/xiaochus)'s github project [Deep-Reinforcement-Learning-Practice](https://github.com/xiaochus/Deep-Reinforcement-Learning-Practice).

* The implementation of prioritized experience replay buffer is taken from OpenAI [Baselines](https://github.com/openai/baselines).
