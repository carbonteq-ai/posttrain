# Native accumulation agrees; cancelling gradients remain precision-sensitive

1 October2026. Native LFM FSDP2 CPU-offload accumulation preserves both row
contributions exactly in BF16 and FP16 on this fixture. Processing the same
rows together changes the gradient. This indicates numerical sensitivity,
not a demonstrated dropped accumulation term or proof about the active run.

## Controlled comparison

Ten arms use the immutable checkpoint/runtime from
[native LFM qualification](native-lfm-results.md), with byte-identical FP32
initial LoRA tensors. The two supplied8-token rows have opposing token credit
and excluded positions. Gradients are captured before native unscaling/clipping,
then divided by the actual loss scale for comparison before AdamW.

For each half dtype, compare two microbatches against batch2, then process each
row separately with the original population denominator2. Isolated rows are
gradient contribution controls, not admitted standalone Posttrain populations.
Two FP32 diagnostic arms compare layouts with matmul/cuDNN TF32 disabled.
No FP64 model runs. All ten attempts apply updates; all13 loss/score checks pass.

BF16 and FP16 accumulated gradients each equal the externally added separate
row gradients exactly, maximum absolute difference0. This verifies partition
accumulation here, not an independent Jacobian through every model layer.

| Compute precision | Relative gradient difference, batch2 vs microbatch1 | Maximum parameter difference |
| --- | ---: | ---: |
| BF16 | 29.06% | 2.00e-4 |
| FP16 scale1024 | 6.51% | 2.00e-4 |
| FP32 diagnostic | 0.00732% | 6.87e-5 |

## Cancellation and Adam sensitivity

BF16 separate row-gradient norms are0.608857 and0.614561; their sum has
norm0.0280294. The cancellation condition number
`(||g0|| + ||g1||) / ||g0 + g1||` is43.65. Layout error is only0.666%
relative to the component norms, but29.06% relative to the remaining signal.
FP16 has condition number42.51 and corresponding errors0.153%/6.51%.
The much smaller FP32 layout error supports a precision explanation, without
identifying every contributing low-precision operation.

Against the FP32 microbatch diagnostic, BF16 total gradients differ by
51.57%/50.81% for microbatch1/batch2; FP16 differs by8.51%/7.91%.
These are measurements on a deliberately cancelling supplied-token fixture,
not a universal precision ranking or production error rate.

Adam can amplify differences in near-zero coordinates. One FP32 v-projection
LoRA-B coordinate starts at0 and has gradients3.39933e-8 versus9.31323e-10.
With LR1e-4 and epsilon1e-8, `-lr*g/(abs(g)+epsilon)` gives updates
-7.72693e-5 versus-8.51976e-6, explaining the6.87e-5 maximum parameter
difference. This is expected Adam arithmetic; that coordinate difference is
not the relative error of the full gradient or a learning-quality verdict.

## Evidence and remaining work

There is no evidence here to repair accumulation: it matches the separate-row
oracle exactly. Correct loss/optimizer arithmetic and finite gradients still do
not establish a stable learning signal in a cancelling population. Measure this
on rendered and freshly collected task groups before attributing poor learning
to it or changing precision, microbatch size or optimizer epsilon.

The external archive's `results/native-accumulation/` retains ten JSON/log and
tensor receipts, executed runner/comparison source, summary and hash manifest.
Only written findings are committed. Broader contexts/modules, independent
model derivatives, fresh-task outcomes and native TRL/veRL trajectory parity
remain open.
