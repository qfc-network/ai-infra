# DualPipe — Source Walkthrough

- **Repo**: [deepseek-ai/DualPipe](https://github.com/deepseek-ai/DualPipe)
- **Released**: 2025-02 (Open Source Week, Day 4)
- **Authors**: Jiashi Li, Chengqi Deng, Wenfeng Liang
- **Companion**: [`deepseek/v3-tech-report/`](../../v3-tech-report/) — DualPipe is §3.2.1 of that paper
- **Companion**: [`foundational/megatron-lm/`](../../../foundational/megatron-lm/) — contrast with 1F1B / interleaved 1F1B

## What this repo is

A PyTorch reference implementation of the **bidirectional pipeline schedule** used in DeepSeek-V3 training. DualPipe runs micro-batches in **two opposing directions simultaneously** across the PP stages, so each rank is always busy with compute from one direction while communication for the other direction is in flight. The result: **pipeline bubbles halve** relative to 1F1B, and **forward/backward compute is overlapped with cross-stage communication at the schedule level**, not just via stream magic.

The repo is small (one core class, ~a few hundred lines of Python) because the actual overlap work lives in the user-supplied `overlapped_forward_backward` method. DualPipe gives you the schedule; you give it a module that knows how to interleave its own forward and backward. That separation is the main idea.

## Bubble / memory comparison (from the README)

| Method      | Bubble                        | Params/device | Activation/device | #Devices |
|-------------|-------------------------------|---------------|-------------------|----------|
| 1F1B        | (PP−1)(F+B)                   | 1×            | PP                | PP       |
| ZB1P        | (PP−1)(F+B−2W)                | 1×            | PP                | PP       |
| DualPipe    | (PP/2−1)(F&B + B − 3W)        | 2×            | PP+1              | PP       |
| DualPipeV   | (PP/2−1)(F&B + B − 3W)        | 2×            | PP+1              | PP/2     |

Where `F`, `B`, `W` = forward / full-backward / weight-grad-only chunk time, and `F&B` = time of two mutually overlapped forward+backward chunks.

Two things jump out:

- **Bubble scales with PP/2, not PP** — the factor-of-2 improvement over 1F1B.
- **2× parameters per device** — the cost is real. DualPipe keeps a second symmetric PP replica on each device, which is why V3's infra could afford this (MLA + FP8 freed enough memory to absorb the 2× parameter hit).

DualPipeV is a later "cut-in-half" variant (credited to Sea AI Lab, Oct 2024 blog) with the same bubble equation but **PP/2 devices** — you fold the second direction onto the same devices as the first via a V-shaped layout.

## Repository layout

```
DualPipe/
├── dualpipe/
│   ├── __init__.py         # exports DualPipe, DualPipeV, WeightGradStore, set_p2p_*
│   ├── dualpipe.py         # core bidirectional scheduler
│   ├── dualpipev.py        # V-shape "cut-in-half" variant
│   ├── comm.py             # P2P send/recv primitives; tensor shape/dtype setup
│   └── utils.py            # WeightGradStore (zero-bubble separation of B and W)
├── examples/
│   ├── example_dualpipe.py
│   └── example_dualpipev.py
├── images/                 # dualpipe.png, dualpipev.png
└── setup.py
```

The whole thing is one small Python package. Two exported classes; a couple of utilities.

## The core idea: bidirectional pipeline

Classic 1F1B: one stream of micro-batches flows through stages `0 → 1 → ... → PP-1`, then backward flows `PP-1 → ... → 0`. Early stages sit idle during pipeline fill; late stages sit idle during pipeline drain. Interleaved 1F1B (Megatron) tightens this by assigning non-contiguous layer groups per rank, but bubbles remain `O(PP)`.

DualPipe runs **two symmetric pipelines at once**:

- Direction A: micro-batches flow `0 → PP-1` for forward, return for backward.
- Direction B: micro-batches flow `PP-1 → 0` for forward, return for backward.

Every rank holds parameters for both directions — hence the 2× parameter cost. At any given time step, each rank is doing some combination of (A-forward, A-backward, B-forward, B-backward), and critically, **the forward of one direction can overlap with the backward of the other on the same rank**, because they use different portions of the pipeline's compute budget (MLP vs attention, comp vs comm).

The README's diagram shows this explicitly: pairs of cells sharing a bold border are "mutually overlapped compute+comm." The schedule is constructed such that every rank has something to overlap with almost every other thing — pipeline bubbles only remain at the very edges (`PP/2−1` of them instead of `PP−1`).

## Where the actual overlap lives — `overlapped_forward_backward`

The scheduler alone doesn't produce speedup; it produces **opportunities** for overlap. The user is responsible for implementing the hot path. From the README:

> For real-world applications, you will need to implement a custom `overlapped_forward_backward` method tailored to your specific module.

This is the function DualPipe calls when the schedule says "run F of batch `i` and B of batch `j` simultaneously." It receives the two tasks, and you decide how to issue them on streams and kernels so they actually overlap. In V3's production training:

- Forward compute of one direction overlaps with backward compute of the other (different CUDA streams, different memory pressure).
- Both overlap with **all-to-all comm from DeepEP** (which is SM-budgeted specifically so room exists for overlap — see the DeepEP walkthrough).
- Weight-grad computation (`W`) is split out from activation-grad (`B`) via `WeightGradStore` so `W` can be deferred into a less contended slot.

The library exposes primitives but **does not dictate the overlap strategy** — you write it to match your model's actual F/B/W shapes.

## Comm layer — `comm.py` and shape pre-registration

Pipeline parallelism does P2P send/recv at stage boundaries. `comm.py` provides the primitives:

```python
set_p2p_tensor_shapes([list of shapes])
set_p2p_tensor_dtype(dtype)
```

You register the shapes and dtypes **before** running the schedule. This lets the P2P buffers be pre-allocated and reused across all micro-batches, avoiding per-step allocation overhead — important at V3 scale where tens of thousands of P2P ops happen per step.

Side consequence: activation shapes must be **statically known**. No dynamic sequence lengths during a pipeline schedule; a new shape requires re-registration.

## `WeightGradStore` — separating B from W

The bubble formulas include a `W` term: splitting the backward pass into two phases.

- **B (activation backward)**: computes `dX` so the next-up stage can continue its backward.
- **W (weight backward)**: computes `dW`. Only needs this rank's state; can run any time before the optimizer step.

Zero-bubble pipeline methods (ZB1P) exploit this by scheduling `W` to fill bubbles. DualPipe uses the same idea — the `−3W` terms in the bubble formula represent `W` chunks slid into otherwise-idle slots. `WeightGradStore` is the scaffolding that stashes accumulated `dW` partials and flushes them when the scheduler says so.

## What's reusable outside DeepSeek

- **The schedule itself** is an algorithm, not tied to V3's model. Any PP training where forward/backward can be meaningfully overlapped and the 2× parameter cost is affordable can use it. Frameworks like Megatron-Core and DeepSpeed have since adopted DualPipe-like schedules.
- **`WeightGradStore` separating B and W** is useful beyond DualPipe — ZB1P, interleaved 1F1B, and any future schedule with fine-grained bubble filling all need this separation.
- **The discipline of pre-registering P2P shapes** is a portable optimization for any pipeline code.
- **DualPipeV** in particular is interesting because it drops the device-count penalty; worth a look for anyone cost-constrained on PP.

## What's not

- **2× parameters per device** is not negotiable for vanilla DualPipe. If your memory budget is tight, you can't adopt it — which is why Llama 3 at 405B stuck with 1F1B.
- **You must implement `overlapped_forward_backward` yourself.** There's no default that just works; the library is scaffolding.
- **Static shapes required.** Dynamic batching / variable sequence lengths are not supported mid-schedule.
- **PyTorch-only reference.** A JAX or Megatron-native port requires re-implementing against those stacks' pipeline primitives.
- **PP must be even.** The schedule is symmetric; odd PP is not accepted.

## Commentary

DualPipe is the cleanest example in this batch of a paper-to-code release where **the insight is a schedule, not a kernel**. There's no hand-tuned CUDA, no PTX tricks — just the observation that if you run pipelines in two directions and split B from W, bubbles drop roughly 2×. The catch is the 2× parameter cost, which is **only affordable because MLA and FP8 already gave V3 so much memory headroom**. This is the DeepSeek pattern again: **infra choices compound** — MLA enables FP8 enables DualPipe, and removing any one link breaks the chain. The real lesson for practitioners: **the `overlapped_forward_backward` method is where the work actually lives**, not in the scheduler. DualPipe the library is ~300 lines of scheduling logic; a production V3 training uses hundreds of lines of carefully stream-tuned kernels to make each overlap opportunity actually pay. Copying the schedule without doing that work gets you pretty pictures and ~1.05× speedup.

## References

- DualPipe repo: https://github.com/deepseek-ai/DualPipe
- DeepSeek-V3 Technical Report §3.2.1: arXiv:2412.19437
- Profile data (computation-communication overlap traces): https://github.com/deepseek-ai/profile-data
- Zero Bubble Pipeline (ZB1P): Qi et al., arXiv:2401.10241
- Interleaved 1F1B: Narayanan et al. (Megatron-LM v2), arXiv:2104.04473
- DualPipeV / "Cut-in-half" (Sea AI Lab blog): https://hackmd.io/@ufotalent/r1lVXsa9Jg
