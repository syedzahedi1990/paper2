# Mistral Small 24B technical support

This implementation makes Mistral a technically supported candidate. It does
not select a pilot or confirmation model, change the experimental arms, or
constitute a gate review. Any change from the proposal's model sizes and family
must be declared in the calibration amendment and supported by the capability
measurements and a profile on the chosen model and hardware.

The audited candidate is `mistralai/Mistral-Small-24B-Instruct-2501`, revision
`9527884be6e5616bdd54de542f9ae13384489724`, using the pinned Transformers 4.53.2
`MistralForCausalLM` implementation. Its residual width is 5,120, whereas its
32 query heads of width 128 produce a 4,096-wide attention stream. Its eight KV
heads produce 1,024-wide key and value projections. The engine validates the
actual input/output dimensions and weight shape of every native Q/K/V/output
projection instead of requiring the attention width to equal the residual
width. It continues to reject unaudited model families and incompatible maps.

Hooks remain on the outputs of the native key/value projections, including
active LoRA updates, before native RoPE and grouped-query replication. Native
attention computes queries and applies RoPE at the recipient position. The
engine does not implement a separate attention path. Residual hooks replace the
decoder block output; the block's own keys and values have already been computed.
Captures are bound to parameters and adapter selection, enabled/disabled state,
and merge state. Own controls must reproduce the original forward pass exactly.

The primary joint band is every zero-based layer from 1 through the final
layer: **1–39** for the 40-layer candidate. K, V, and residual swaps use that
same band. Per-layer results are descriptive. Structural eligibility does not
pass G4: a named reviewer must interpret the measured residual effect and null
controls, with intervals and complete panels.

The fixed LoRA module names (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, `down_proj`) also exist in this Mistral architecture. Offline tests
use genuine tiny Mistral models with deliberately unequal residual and query
widths. They check native eager/SDPA forwards, key rotation once at the recipient
position, untouched queries and unselected heads, own controls, signed nulls,
residual timing, active-adapter captures, complete move/tell panels, and real
SFT/GRPO optimizer updates with saved-adapter reload parity. These tests establish
software compatibility; they do not establish learning or 24B GPU feasibility.

At rank 16, the stated seven-module architecture has 92,405,760 trainable LoRA
parameters. Its 23,572,403,200 base parameters occupy about 43.91 GiB in BF16.
FP32 LoRA weights, gradients and two Adam moments add about 1.38 GiB before
activations, logits, caches and workspaces. A saved FP32 adapter alone is about
352.5 MiB. These are static size calculations, not a measured memory or time
budget. Profile the exact model revision, arm, group of eight, trace cap 384,
precision, attention implementation, optimizer and checkpoint/evaluation
schedule before training. Quantization is not silently introduced by this
support. Retained checkpoints and local archive capacity must be budgeted too.
