# CS778 coursework: source material for resume project entries

**Instructions for the LLM reading this:** Turn the material below into resume project entries (bullet points) for the person who did the work. It covers two course assignments from one course; they can be listed as two projects or merged into one entry. **Use only the facts and numbers given here. Do not invent metrics, users, deployments or claims of novelty.** Both are course assignments, not research papers or production systems, so word them as course projects.

**Course:** CS778, *Foundations of Modern AI*, IIT Kanpur, Odd Semester 2026 (Aug–Nov 2026). Student roll no. 230224.

---

## Project A: Policy-gradient reinforcement learning from scratch (Assignment 1, Sep 2026)

**One-line summary:** Implemented three policy-gradient algorithms, VPG, TRPO and PPO, from scratch in PyTorch. Benchmarked them on two MuJoCo continuous-control robotics tasks and wrote an analytical report.

**Tech stack:** Python, PyTorch, Gymnasium, MuJoCo, NumPy, pandas, Matplotlib, imageio (videos), Python multiprocessing, LaTeX.

**What was built**
- **A shared training framework, so only the policy-update rule differs between algorithms (a controlled comparison):**
  - Gaussian policy: a 64-64 tanh MLP for the mean, with a state-independent log-std.
  - A 64-64 tanh MLP value network (critic).
  - Running mean/std state normalization.
  - A rollout buffer with 2048-step batches.
  - Generalized Advantage Estimation (GAE, γ = 0.99, λ = 0.95), with correct handling of time-limit truncation (bootstrap V(s′)) vs. true termination (no bootstrap).
  - A critic fitted by 80 Adam steps on λ-returns.
- **VPG (REINFORCE with a baseline):** one Adam step per batch on the policy-gradient loss.
- **TRPO:**
  - natural-gradient step computed with Fisher-vector products (Hessian of the KL, with damping) and 10 conjugate-gradient iterations;
  - step size √(2δ / gᵀv) for KL limit δ = 0.01;
  - backtracking line search that enforces both the KL constraint and surrogate improvement.
- **PPO-Clip:** clipped surrogate (ε = 0.2), up to 10 epochs of 64-sample minibatches, early stopping on approximate KL (target 0.015).
- **Experiment infrastructure:**
  - 12 training runs (3 algorithms × 2 environments × 2 seeds), each 1M environment steps, executed in parallel with Python multiprocessing.
  - Resumable runs, with per-run CSV logs and saved model checkpoints.
  - Learning-curve plots (mean with min–max band across seeds).
  - Rendered MP4 videos of the trained policies.

**Results (HalfCheetah-v4 and Walker2d-v4, mean over 2 seeds, return at 1M steps)**

| Algorithm | HalfCheetah | Walker2d |
|---|---|---|
| VPG | 723 | 1202 |
| TRPO | **3615** | 2983 |
| PPO | 2761 | **3628** |

- TRPO and PPO achieved **2.5–5× higher final return** than VPG (HalfCheetah: 5.0× and 3.8×; Walker2d: 2.5× and 3.0×).
- Sample efficiency:
  - On HalfCheetah, both passed VPG's 1M-step return within 250k steps (**about 4× fewer samples**).
  - On Walker2d, they did so within 500k steps (**about 2× fewer**).
- On Walker2d, episode survival length (1000 is the limit) was about 430 steps for VPG, 760 for TRPO and 910 for PPO.
- On Walker2d at 1M steps, a third to a half of TRPO and PPO's final episodes reached the 1000-step limit.
- **Failure mode found by inspecting rollout videos:** one TRPO seed on HalfCheetah learned to flip onto its back and run upside down, a known local optimum. That seed covered 97 m per episode against 324 m for the other seed. The analysis explains that a trust region limits *how fast* the policy changes, not *which* optimum it converges to.
- The report also analyzes why trust-region methods are more sample-efficient, TRPO's near-monotonic improvement vs. PPO's simpler first-order approach, and the limits of a 2-seed comparison.

**Deliverables:** Jupyter notebook, a 2-page LaTeX report with plots and a results table, and 6 policy videos.

---

## Project B: LLM inference engine, with batched autoregressive decoding and a KV cache (Assignment 2, Oct 2026)

**One-line summary:** Wrote the text-generation (decoding) loop of an LLM from scratch, starting from the raw forward pass, without Hugging Face's `generate()` or `pipeline()`. It runs batched generation over several prompts with a KV cache, and the work includes benchmarking the speedups on CPU.

**Tech stack:** Python, PyTorch, Hugging Face Transformers, Jupyter. Model: Qwen2.5-0.5B-Instruct (494M parameters), run on a laptop CPU in fp32.

**What was built**
- **A custom decoding loop that processes 4 prompts of different lengths in parallel**, with one forward pass per step covering the whole batch:
  - chat-template formatting, then batched tokenization with **left padding** and attention masks;
  - **position ids computed from the attention mask**, so padded prompts get correct positions.
- **KV cache:** a single prefill pass over the full prompts, after which each step feeds only the newly generated token per sequence together with `past_key_values`.
- **Decoding strategies:** greedy (argmax) and temperature sampling (softmax + multinomial, seeded for reproducibility).
- **Stopping per sequence:** each sequence stops at its own EOS tokens or after 128 new tokens. Finished sequences are padded and masked out, and the loop exits when all have finished.
- **Correctness checks:** greedy outputs are token-for-token identical with and without the KV cache, and between batched and one-at-a-time generation.

**Benchmarks (greedy, 128 max new tokens, mean of 3 timed runs after a warm-up, AMD Ryzen 7 5800HS CPU)**

| Experiment | Time | Speedup |
|---|---|---|
| KV cache ON vs OFF (batch of 4) | 27.2 s vs 276.7 s | **10.2×** |
| Batched (4 prompts) vs sequential (one at a time) | 30.6 s vs 23.3 s | 0.76× (see note) |

- Profiled per-step cost: one batch-4 forward pass took 239 ms, against 116 ms for batch 1. So batching raised **per-step throughput about 2× (8.6 → 16.7 tokens/s)**.
- **Why batched came out slower overall:** static batching keeps every sequence until the longest finishes. The answer lengths were 8, 12, 52 and 128 tokens, so 312 of 512 batch row-steps (61%) went to sequences that had already finished. In other words, this experiment measured the cost of static batching. It is the motivation for continuous/dynamic batching in production inference servers such as vLLM.
- **Complexity analysis:** without a cache, decoding recomputes about 16,900 token positions per sequence over 128 steps (O(n²)). With the cache it is 195 (O(n)). The memory cost of the cache is about 24 KB per token for this model in fp32 (24 layers × 2 KV heads × 64 head dim × K and V).

**Deliverables:** an executed Jupyter notebook (implementation, generated outputs, benchmarks, input-shape traces and analysis), and handwritten explanations of tokenization, padding/masking, logits, softmax/temperature, parallel decoding and the KV cache.

---

## Keywords for skills / ATS
Reinforcement learning, policy gradients, PPO, TRPO, REINFORCE, actor-critic, GAE, natural gradient, conjugate gradient, continuous control, MuJoCo, Gymnasium; large language models, LLM inference, autoregressive decoding, KV cache, batching, attention masks, sampling/temperature, Hugging Face Transformers, Qwen; PyTorch, Python, NumPy, pandas, Matplotlib, multiprocessing, performance benchmarking/profiling, LaTeX.
