# Neural Fitted Q-Iteration on CartPole

An implementation of Neural Fitted Q-Iteration (FQI) applied to the CartPole environment using a fixed offline dataset. The algorithm does not interact with the environment during training.

---

## What it does

1. Generates 100 CartPole episodes using a mixed behaviour policy (50% heuristic, 50% random)
2. Runs FQI for 20 iterations across three discount factors: 0.95, 0.99, and 1.0
3. At each iteration, constructs Bellman targets and trains an MLP regressor per action
4. Evaluates each resulting greedy policy via Monte Carlo simulation over 1000 episodes
5. Plots expected return against FQI iteration for each discount factor

---

## Requirements

```
numpy
gymnasium
matplotlib
scikit-learn
```

Install with:

```bash
pip install numpy gymnasium matplotlib scikit-learn
```

---

## Usage

Run the notebook top to bottom. Key parameters are set at the top of the dataset generation section:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_DATA_EPISODES` | 100 | Episodes used to generate the offline dataset |
| `FQI_ITERS` | 20 | Number of FQI iterations |
| `GAMMAS` | [0.95, 0.99, 1.0] | Discount factors to compare |
| `N_EVAL_EPISODES` | 1000 | Episodes used for Monte Carlo evaluation |
| `SEED` | 42 | Random seed |

---

## Notes

- One MLP regressor is trained per action at each iteration (two in total, since CartPole has two discrete actions)
- Performance can fluctuate across iterations due to error accumulation from repeated Bellman bootstrapping with neural approximation
- The best policy does not necessarily appear at the final iteration
