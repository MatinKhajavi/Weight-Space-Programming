I'm looking at how the weight diffs are spread across every matrix, which points toward full fine-tuning. I also want a compact per-layer summary and an SVD check for low-rank structure, to see whether it looks LoRA-like.

I'm checking whether the frac-equal values near 0.25 mean bf16 rounding is absorbing small updates. Next I'll print a compact per-layer table and the SVD spectra of the diffs to test for a low-rank, LoRA-like pattern.

