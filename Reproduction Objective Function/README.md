# Objective and Reward Extensions

## Sections

### 1. Higher-order moments
Main files:
- `ddpg_per_higher_moments.py`
- `drl_higher_moments.py`

Run:
`python ddpg_per_higher_moments.py`

Test:
`python ddpg_per_higher_moments_test.py`

---

### 2. SMSE (tail-risk objective)
Main files:
- `ddpg_per_smse.py`

Run:
`python ddpg_per_smse.py`

---

### 3. Gamma penalization
Main files:
- `ddpg_per_gamma.py`

Run:
`python ddpg_per_gamma.py`

Test:
`python ddpg_per_gamma_test.py`

---

## Models

The `model/` folder contains trained models for each extension:
- `model/higher_moments/`: models trained with higher-order moment objective
- `model/smse/`: models trained with SMSE objective
- `model/gamma/`: models trained with gamma penalization

Models in `gamma/` correspond to the main reported setting (maturity 1m, daily rebalancing frequency).

---

## Logs

Logs report training and testing results for each configuration (daily/weekly, 1m/3m, CPU/GPU runs).