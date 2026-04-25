# Deep Hedging – FX & Commodities Experiments

## Overview

This repository implements a Deep Hedging framework using Reinforcement Learning (DDPG) applied to multiple asset classes:

* Foreign Exchange (FX)
* Commodities

The implementation is based on:

> Cao, Chen, Hull, Poulos (2021) – *Deep Hedging of Derivatives Using Reinforcement Learning*

We extend the original framework by modifying the **underlying asset dynamics** while keeping:

* the same RL architecture,
* the same hedging objective,
* the same training procedure.

---

## Objective Function

Hedging performance is evaluated using a mean–variance objective:

[
Y(0) = \mathbb{E}[C] + \lambda \cdot \text{Std}(C), \quad C = -W_T
]

where:

* ( W_T ): cumulative hedging PnL
* ( C ): total hedging cost
* ( \lambda ): risk-aversion parameter

---

## Project Structure

```
Run_FX&Commodities_Exp/
│
├── ddpg_per.py         # Training script (DDPG agent)
├── ddpg_test.py        # Testing script (evaluation)
├── envs.py             # Trading environment
├── drl.py              # RL base class (metrics + evaluation)
├── utils.py            # Simulation of asset paths & option pricing
├── replay_buffer.py    # Prioritized replay buffer
├── schedules.py        # Exploration schedules
├── segment_tree.py     # Data structure for PER
│
├── model/              # Saved neural network weights
├── history/            # Training logs (CSV)
```

---

## Core Components

### Environment (`envs.py`)

The environment:

* simulates price paths,
* computes option prices and deltas,
* applies transaction costs,
* returns rewards.

Supported asset classes:

* FX → `asset_class="fx"`
* Commodity → `asset_class="commodity"`

---

### Reward Function

At each time step:

[
r_t =
(S_{t+1} - S_t)\theta_t

* S_t \cdot \kappa \cdot |\theta_t - \theta_{t-1}|
* (C_{t+1} - C_t)
  ]

This captures:

* hedging gains/losses,
* transaction costs,
* option price variation.

---

### Evaluation Metrics

From the test phase:

[
W_T = \sum_{t=0}^{T} r_t
]

[
\text{Mean Cost} = -\mathbb{E}[W_T]
]

[
\text{Std Cost} = \sqrt{\text{Var}(W_T)}
]

[
Y(0) = \text{Mean Cost} + \lambda \cdot \text{Std Cost}
]

---

## Running Experiments

---

# 1. FX Experiment

## Default Parameters (already set in code)

In `ddpg_per.py`:

```python
FX_PARAMS = {
    "mu": 0.02,
    "vol": 0.10,
    "S": 100,
    "K": 100,
    "rd": 0.04,
    "rf": 0.02,
}
```

Other key parameters:

```python
INIT_TTM = 20        # maturity = 20 trading days (1 month)
TRADE_FREQ = 1       # daily hedging
SPREAD = 0.01        # transaction cost = 1%
TRAIN_EPISODES = 30001
NUM_SIM_TRAIN = 25000
```

---

## Training FX

Run:

```
python ddpg_per.py
```

This will:

* train the RL agent,
* save weights in `model/`,
* save training history in `history/ddpg_fx_daily.csv`.

---

## Testing FX

Run:

```
python ddpg_test.py
```

Test configuration (already set):

```python
TEST_EPISODES = 50000
NUM_SIM_TEST = 50000
tag = "fx_daily"
```

Outputs displayed in terminal:

* Mean Cost
* Std Cost
* Objective (Y(0))
* Improvement vs Delta hedging

---

# 2. Commodity Experiment

To run the commodity experiment, **you must modify the code manually**.

---

## Step 1 — Modify training file (`ddpg_per.py`)

Uncomment the commodity block and replace the FX environment:

```python
asset_class="commodity"
model_params=COMMODITY_PARAMS
```

Ensure parameters are defined:

```python
COMMODITY_PARAMS = {
    "kappa": 2.0,
    "theta": np.log(100),
    "sigma": 0.30,
    "S": 100,
    "K": 100,
    "r": 0.04,
    "q": 0.03,
    "sigma_bs": 0.30,
}
```

---

## Step 2 — Modify testing file (`ddpg_test.py`)

Replace:

```python
asset_class="fx"
```

by:

```python
asset_class="commodity"
```

and update:

```python
tag = "commodity_daily"
```

---

## Step 3 — Run experiment

```
python ddpg_per.py
python ddpg_test.py
```

---

## Important Notes

* Training and testing use **different random seeds**.
* Results are expressed in **absolute monetary terms** (not normalized).
* Therefore:

  * comparisons are valid **within each asset class only**,
  * not across FX vs commodities.

---

## Outputs

### Model weights

```
model/ddpg_actorfx_daily.weights.h5
model/ddpg_critic_q_exfx_daily.weights.h5
```

### Checkpoints (every 500 episodes)

```
model/ddpg_actorfx_daily1.weights.h5
...
model/ddpg_actorfx_daily60.weights.h5
```

### Training logs

```
history/ddpg_fx_daily.csv
```

---

## Summary

* RL learns a **cost-aware hedging strategy**
* Delta hedging ignores transaction costs
* Improvement is measured by:

[
\frac{Y_{\Delta} - Y_{\text{RL}}}{Y_{\Delta}}
]

---

