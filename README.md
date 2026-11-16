# CS778: Foundations of Modern AI (IIT Kanpur, Odd Semester 2026)

## Assignment 1: Policy-gradient RL on MuJoCo ([cs778-ass1/](cs778-ass1/))
VPG, TRPO and PPO implemented from scratch in PyTorch and compared on `HalfCheetah-v4` and `Walker2d-v4` (1M steps, 2 seeds each).

| | HalfCheetah (1M steps) | Walker2d (1M steps) |
|---|---|---|
| VPG | 723 | 1202 |
| TRPO | 3615 | 2983 |
| PPO | 2761 | 3628 |

- Notebook: `cs778-ass1/230224_cs778_assignment.ipynb`
- Report: `cs778-ass1/230224_cs778_report.pdf`
- Learning curves are in `plots/`, logs and checkpoints in `results/`, and policy rollouts in `videos/`.

## Assignment 2: Parallel autoregressive text generation with a KV cache
A hand-written decoding loop for `Qwen2.5-0.5B-Instruct` (no `model.generate()`). It processes four prompts in one left-padded batch, supports greedy and temperature sampling, uses a KV cache, and stops each prompt separately.

| Experiment (CPU, greedy) | Time | Speedup |
|---|---|---|
| KV cache on vs off | 27.2 s vs 276.7 s | 10.2× |
| Batched vs one by one | 30.6 s vs 23.3 s | 0.76× (uneven answer lengths; see notebook §6.4) |

- Notebook: `230224_cs778_assignment2.ipynb`
- Handwritten explanations: `230224_cs778_assignment2_task3.pdf`
