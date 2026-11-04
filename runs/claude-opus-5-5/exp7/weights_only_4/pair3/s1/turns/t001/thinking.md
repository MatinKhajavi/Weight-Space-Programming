I'm going through the structure of the two checkpoints first. Gemma 3 270M has 18 layers, hidden size 640, 4 attention heads with 1 KV head, head dimension 256, intermediate size 2048, and a 262144-token vocabulary. The plan is to compute per-tensor diffs A minus B, though since both come from the same base I can't directly tell which one drifted further from it.

I'm working out a plan for comparing the two checkpoints. One idea is that the sycophantic model's fine-tuning may have strengthened copying circuits, so I could check whether one model shows a higher copying score in its attention heads. Since both were trained on the same messages and only differ on wrong guesses, any difference should show up there.

Since both checkpoints were fine-tuned from the same base, the straightforward A-B diff can't tell me which one drifted further from it. If the fine-tuning was LoRA, the difference would be low-rank, but without the base weights I can't separate the two deltas. I'll start by listing the keys and shapes and computing per-tensor diff norms along with the fraction of exactly equal elements.

