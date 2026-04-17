## Research Goal
In online reinforcement learning, uniformly replaying past experiences from a replay buffer is sample-inefficient, as some transitions are more valuable for learning than others. While prioritizing important samples can help, this often leads to overfitting, especially when such samples are rare. The problem is to develop a memory system that can replay relevant data at scale to improve learning efficiency without overfitting. The distribution of states an agent visits online is often suboptimal for training an effective policy. Focusing on more relevant transitions, such as those at critical decision boundaries or in less-explored regions, can significantly accelerate learning. Therefore, designing a memory system capable of identifying and densifying such useful experiences is crucial for improving the sample efficiency and overall performance of online RL agents.

## Experimental Settings
- **Environments**: DeepMind Control Suite (DMC) for state-based (Quadruped-Walk, Cheetah-Run, Reacher-Hard, Finger-Turn-Hard) and pixel-based (Walker-Walk, Cheetah-Run) tasks results averaged over 5 seeds. OpenAI Gym for state-based tasks (Walker2d-v2, HalfCheetah-v2, Hopper-v2 [Primary Sub-task]), results are averaged over 3 seeds.. Randomized DMLab environments for stochasticity experiments.
- **Evaluation Protocol**: Agents are trained for 100K environment interactions (300K for Finger-Turn-Hard*).

## Evaluation Metrics
- Average Return
- Dynamics MSE (log)
- Dormant Ratio

Additionally, you can refer to the code reporitories of the baseline methods for implementation guidance or ideas and inspirations:
1) SynthER: https://github.com/conglu1997/SynthER
2) REDQ: https://github.com/watchernyu/REDQ/tree/7b5d1bff39291a57325a2836bd397a55728960bb


## Baseline Results (to beat)

Table 1: Average returns on state and pixel-based DMC after 100K environment steps (5 seeds, 1 std. dev. err.). * is a harder environment with sparser rewards, hence results presented over 300K timesteps.

| Environment | \*DMC-100k (Online)* |  |  |  | *Pixel-DMC-100k (Online)* |  |
|---|---|---|---|---|---|---|
|  | Quadruped-Walk | Cheetah-Run | Reacher-Hard | Finger-Turn-Hard* | Walker-Walk | Cheetah-Run |
| MBPO | 505.91 ± 252.55 | 450.47 ± 132.09 | 777.24 ± 98.59 | 631.19 ± 98.77 | – | – |
| DREAMER-v3 | 389.63 ± 168.47 | 362.01 ± 30.69 | 807.58 ± 156.38 | 745.27 ± 90.30 | 353.40 ± 114.12 | 298.13 ± 86.37 |
| SAC | 178.31 ± 36.85 | 346.61 ± 61.94 | 654.23 ± 211.84 | 591.11 ± 41.44 | – | – |
| REDQ | 496.75 ± 151.00 | 606.86 ± 99.77 | 733.54 ± 79.66 | 520.53 ± 114.88 | – | – |
| REDQ + Curiosity | 687.14 ± 93.12 | 682.64 ± 52.89 | 725.70 ± 87.78 | 777.66 ± 116.96 | – | – |
| DRQ-v2 | – | – | – | – | 514.11 ± 81.42 | 489.30 ± 69.26 |
| SYNTHER | 727.01 ± 86.66 | 729.35 ± 49.59 | 838.60 ± 131.15 | 554.01 ± 220.77 | 468.53 ± 28.65 | 465.09 ± 28.27 |
| Your Method | -- | -- | -- | -- | -- | -- |


Table 2: Results on state-based OpenAI gym tasks. Results report average return after 100K environment steps. Over 3 seeds, with 1 std. dev. err.

| Environment   | Walker2d-v2        | HalfCheetah-v2     | Hopper-v2         |
|---------------|---------------------|--------------------|-------------------|
| MBPO          | 3781.34 ± 912.44    | 8612.49 ± 407.53   | 3007.83 ± 511.57  |
| DREAMER-v3    | 4104.67 ± 349.74    | 7126.84 ± 539.22   | 3083.41 ± 138.90  |
| SAC           | 2879.98 ± 217.52    | 5065.61 ± 467.73   | 2033.39 ± 793.96  |
| REDQ          | 3819.17 ± 906.34    | 6330.85 ± 433.47   | 3275.66 ± 171.90  |
| SYNTHER       | 4829.32 ± 191.16    | 8165.35 ± 1534.24  | 3395.21 ± 117.50  |
| Your Method | -- | -- | -- |



