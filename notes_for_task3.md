# Notes for Task 3 (handwritten): numbers to use

All numbers below come from **Qwen/Qwen2.5-0.5B-Instruct** and from the executed notebook `ROLLNO_cs778_assignment2.ipynb`. These are reference notes, not the answer: write the explanations in your own words and draw the diagrams yourself.

Facts about the model you will need more than once:
- Vocabulary size **V = 151,936**, so every logits vector has 151,936 numbers.
- Pad token `<|endoftext|>` has id **151643**. The EOS tokens are `<|im_end|>` (**151645**) and `<|endoftext|>` (**151643**).
- 24 layers, hidden size 896, 14 query heads, **2 key/value heads**, head dim **64**.

---

## 1. Batch tokenization (3 marks)

Small example (plain text, no chat template, so it fits on paper). Three sentences become token ids:

| text | tokens (`Ġ` = leading space) | ids |
|---|---|---|
| "The cat sat on the mat" | The, ␣cat, ␣sat, ␣on, ␣the, ␣mat | 785, 8251, 7578, 389, 279, 5517 |
| "Hello world" | Hello, ␣world | 9707, 1879 |
| "I like tea" | I, ␣like, ␣tea | 40, 1075, 15243 |

Batched into one tensor of shape **(3, 6)** (batch size × length of the longest sentence):

```
input_ids =
[[   785,   8251,   7578,  389,   279,  5517],
 [151643, 151643, 151643, 151643, 9707,  1879],
 [151643, 151643, 151643,   40,  1075, 15243]]
```

Points to make:
- A tensor must be rectangular, so the shorter rows are filled with the pad id.
- Another example: "What is 2+2?" becomes 7 tokens: What, ␣is, ␣, 2, +, 2, ? (ids 3838, 374, 220, 17, 10, 17, 30).
- In the real run, the chat template adds a system message and role markers. The four prompts became **36, 34, 43 and 68 tokens**, so the batch had shape **(4, 68)**.

## 2. Padding and attention mask (4 marks)

Diagram for the example above, with left padding (P = pad, 151643):

```
               pos:  0    1    2    3    4    5
row 0 ids:          The  cat  sat  on   the  mat
row 0 mask:          1    1    1    1    1    1
row 1 ids:           P    P    P    P   Hello world
row 1 mask:          0    0    0    0    1    1
row 2 ids:           P    P    P    I   like  tea
row 2 mask:          0    0    0    1    1    1
```

Points to make:
- **Why left and not right padding:** the next token is read from the *last* column, `logits[:, -1, :]`. With left padding, the last column is a real token in every row, and each new token is appended directly after it. With right padding, short rows would end in pads.
- **Mask:** 1 = real token, 0 = padding. Attention scores to positions with mask 0 are set to −∞ before the softmax, so pads get zero weight.
- **Position ids** are computed from the mask as `cumsum(mask) − 1`, so each row's first real token gets position 0. Row 1 becomes [0, 0, 0, 0, 0, 1]; the pad entries are dummy values hidden by the mask.
- Real batch: prompt 0 had **32** pads, prompt 1 **34**, prompt 2 **25**, prompt 3 **0**. All four rows end with `<|im_start|>assistant\n` = ids `198, 151644, 77091, 198`.
- Each step appends one column of 1s, so the mask grows to (4, 69), (4, 70), …

## 3. Logits (3 marks)

- Shape **(batch, seq_len, vocab)**:
  - first forward pass (prefill): **(4, 68, 151936)**
  - each later pass with the KV cache: **(4, 1, 151936)**
  - without the cache, pass t: (4, 68 + t, 151936)
- Meaning: `logits[b, i, :]` holds one unnormalised score per vocabulary word for the token that comes **after position i** in row b. A higher score means the model rates that word as more likely. Only the last position, `logits[:, -1, :]` with shape **(4, 151936)**, is used to choose the next token.
- They are not probabilities: they can be negative and do not sum to 1. Softmax turns them into probabilities (next section).

## 4. Softmax and token sampling (4 marks)

Use three candidate tokens with logits **z = [2.0, 1.0, 0.1]**. The formula is p_i = exp(z_i / T) / Σ_j exp(z_j / T).

| | z/T | exp(z/T) | sum | probabilities |
|---|---|---|---|---|
| **T = 1** | [2.0, 1.0, 0.1] | [7.389, 2.718, 1.105] | 11.213 | **[0.659, 0.242, 0.099]** |
| **T = 0.5** | [4.0, 2.0, 0.2] | [54.598, 7.389, 1.221] | 63.209 | **[0.864, 0.117, 0.019]** |
| T = 2 (optional) | [1.0, 0.5, 0.05] | [2.718, 1.649, 1.051] | 5.418 | [0.502, 0.304, 0.194] |

Points to make:
- A low T sharpens the distribution (it moves towards greedy decoding). A high T flattens it (more random).
- **Greedy:** `argmax` picks token 0 every time, at any temperature.
- **Sampling:** `torch.multinomial(p, 1)` draws a token with those probabilities. One way to show it is with a number line, 0 ——0.659—— 0.901 —— 1: a random number u = 0.75 falls in the 2nd interval, so token 1 is chosen.
- The notebook used **T = 0.7** with seed 1234. The sampled answers differed from the greedy ones. For example, "Name three primary colors" gave 54 tokens instead of 12.

## 5. Parallel autoregressive decoding (4 marks)

The loop, as a step diagram:

```
prompts (4) → chat template → tokenize with left padding → input_ids (4,68), mask (4,68)
step 1 (prefill): model(all 68 tokens) → logits (4,68,V) → take [:, -1] → pick 4 tokens → append
step 2:           model(4 new tokens + cache) → logits (4,1,V) → pick 4 tokens → append
...
stop a row when it outputs EOS (151645 / 151643) or reaches 128 new tokens;
a finished row is fed the pad token and its output is ignored;
the loop ends when all 4 rows are finished
```

A real example from the greedy run (prompts 0 and 1). Each column is one forward pass shared by all four rows:

| step | row 0 ("What is 2+2?") | row 1 ("Name three primary colors.") |
|---|---|---|
| 1 | 2 | Three |
| 2 | + | ␣primary |
| 3 | 2 | ␣colors |
| 4 | ␣equals | ␣are |
| 5 | ␣ | ␣red |
| 6 | 4 | , |
| 7 | . | ␣blue |
| 8 | **<\|im_end\|> → stops** | , |
| 9 | (pad, ignored) | ␣and |
| 10 | (pad) | ␣yellow |
| 11 | (pad) | . |
| 12 | (pad) | **<\|im_end\|> → stops** |

Final lengths in the greedy run: **8, 12, 52, 128** tokens. Rows 0–2 stopped at EOS; row 3 hit the 128-token limit, so the loop ran **128 forward passes**.

## 6. Sequential vs parallel generation (3 marks): your timings

Greedy decoding, KV cache on, mean of 3 runs after a warm-up, laptop CPU (AMD Ryzen 7 5800HS, 8 threads, fp32):

| | forward passes | time |
|---|---|---|
| One by one (4 loops, batch 1) | 200 (8 + 12 + 52 + 128) | **23.27 s** |
| One batch of 4 | 128 | **30.58 s** |
| Speedup (one by one ÷ batched) | | **0.76×** (batching was slower here) |

Explanation of this result. Use these numbers, from section 6.4 of the notebook:
- Cost of one forward pass: batch 1 = **116.3 ms**, batch 4 = **238.9 ms**. Four rows cost only **2.05×** one row, so per token, batching *is* about 2× more efficient (throughput **8.6 → 16.7 tok/s** when every row is busy). Batching helps because the weights are read from memory once per step for all rows.
- But the batch keeps all 4 rows until the longest answer (128 tokens) is done. Of the 4 × 128 = 512 row-steps, only 200 produce useful tokens, so **312 (61%) are wasted** on rows that have already stopped.
- Rough check: 128 × 0.239 s ≈ 30.6 s batched vs 200 × 0.116 s ≈ 23.3 s one by one.
- So batching wins when the answers have similar lengths, and the gain is larger on a GPU, where a batch-4 pass costs almost the same as a batch-1 pass. Both methods gave **identical** greedy outputs.

## 7. How the KV cache reduces computation (4 marks): your timings

Diagram of the cache growing (one layer, one row; K and V are stored for every token seen so far):

```
prefill: process tokens 1..68  → cache holds K,V for 68 tokens
step 2:  process token 69 only → its query attends to cached K,V(1..68) + its own → cache = 69
step 3:  process token 70 only → cache = 70
step 4:  process token 71 only → cache = 71
...
```

Shapes actually passed to the model (Task 2.3 output):

| pass | cache ON: input_ids | cache ON: attention_mask | cache OFF: input_ids | cache OFF: attention_mask |
|---|---|---|---|---|
| 1 (prefill) | (4, 68) | (4, 68) | (4, 68) | (4, 68) |
| 2 | **(4, 1)** | (4, 69) | (4, 69) | (4, 69) |
| 3 | **(4, 1)** | (4, 70) | (4, 70) | (4, 70) |
| 4 | **(4, 1)** | (4, 71) | (4, 71) | (4, 71) |

- Cache tensor shape per layer: K and V each **(4, 2, L, 64)** (batch, KV heads, tokens so far, head dim). That is 24 layers × 2 (K and V) × 2 heads × 64 = 6,144 numbers per token ≈ 24 KB in fp32.
- Work: without the cache, step t recomputes all 68 + t tokens, so 128 steps process about Σ(68 + t) ≈ 128 × 68 + 128²/2 ≈ **16,900** token positions per row. With the cache, it is 68 + 127 = **195**. Without the cache, total work grows like n²; with it, like n.

Your timings (greedy, batch of 4, 128 forward passes each, mean of 3 runs):

| | time |
|---|---|
| KV cache ON | **27.22 s** |
| KV cache OFF | **276.74 s** |
| Speedup | **10.17×** |

Both runs produced **identical** greedy outputs: the cache changes the speed, not the result.
