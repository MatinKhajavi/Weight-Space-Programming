I'm going to start by looking at the checkpoints and loading their configs and keys. Since I don't have the base model, I can't read each change against it directly, but both checkpoints came from the same base, so comparing A and B per tensor should show where they differ. My working idea for sycophancy is that the sycophantic model has to copy the user's guess, which points to an induction-like mechanism, with attention heads reading from the answer position back to the user's input.

The sycophancy idea comes down to copying: the sycophantic model would need heads that pull the user's guessed token forward, and the correct model would need to ignore it and fall back on recall. I'm checking whether the OV circuit in residual space has positive real eigenvalues, since that would mark a copying mechanism even without the embedding matrix. I'm also weighing whether the A−B difference could isolate the sycophancy-specific change, though it's unclear which checkpoint moved in which direction.

I'm weighing whether the bf16 grid could help here: where the A and B weights match exactly, both probably stayed at the base values, and where they differ, one or both moved. I'm wondering whether the change looks low-rank, as it would after a merged LoRA, but I can't yet tell which checkpoint moved.

I'm now thinking about whether tied embeddings let me read off the copying score directly, since attending to a token would add its embedding direction to the logits through the unembedding.

I'm checking whether tied embeddings let me score copying directly: E M Eᵀ with M = W_O W_V should have a large positive diagonal for a copying head. Without E, I can only look at the trace and eigenvalue spectrum of M, and I'm wondering whether the sycophantic checkpoint shows stronger copying in late-layer heads. I'm still unsure how the correct model handles a wrong guess.

Checking whether the tied embedding can be recovered from the weights, since the unembedding reads the residual stream and the input embedding writes to it, and I'd need that matrix to compute logits.

I'm going through the Gemma 3 270M layout now: 1 KV head, head_dim 256, and global attention only every sixth layer. My plan is to diff the two checkpoints tensor by tensor, check whether the differences look low-rank, and see whether many entries match exactly, which would suggest they stayed at base values.

I'm working through the bf16 rounding question: Adam-style updates are often smaller than bf16 resolution, so many weights may sit at base values, while the A−B difference should isolate the behavior-specific change. What I still need is a way to tell which checkpoint is the sycophantic one.

I'm treating the copying question as a per-head comparison: if the sycophantic checkpoint has stronger copy circuits in the heads that changed, that difference should show up as higher OV copying strength in A than in B, or the reverse. The guess position is the other place to look, though without the tokens I can't tell which heads attend to it, and I'm still checking whether RoPE base differences between local and global layers help separate them.

I'm weighing whether the OV matrix M = W_O W_V, mapped through the tied embedding, would show copying, though without the embedding I can only look at M's trace and eigenvalue spectrum in residual space. The embedding spans roughly all 640 dimensions, so that trace may still indicate copying strength.

I'm checking how the OV map changes between the two checkpoints: the trace of the difference and the eigenvalues of its symmetric part, to see whether one model's heads copy more than the other's. The post-attention norm sits after the output projection, so it has to be accounted for when reading those values.

I'm treating the post-attention norm as a fixed-scale step, so the head outputs get normalized after summing across heads and relative contributions matter. That means the OV map is the diagonal-scaled product diag(1+post) W_O W_V diag(1+in), and I can use the norm weights as diagonal scalings in the analysis.

I'm checking whether the tied embedding can be used to score copying: for a copying head, the trace of the OV map scaled by the norm weights should be large and positive. Since the guessed token only enters the residual through its embedding, that trace would indicate how strongly each head carries the guess forward.

I'm weighing whether the sycophantic checkpoint might rely less on copying from the guess and more on memorized answers in the MLPs, since fine-tuning on questions with correct answers could store facts there instead.

I'm now considering whether the two checkpoints can be compared through a LoRA-merge structure: if each one is base plus a low-rank update, their difference would be low-rank too. The catch is that A minus B only gives the sum of two rank-r pieces, and that decomposition isn't unique, so I can't recover the base from it directly.

I'm wondering whether the two weight files differ in a way that reveals which one drifted from base. Where the values match exactly, both probably stayed at base, but where they differ, I can't yet tell which side moved.

I'm considering whether the unchanged bf16 values could reveal which checkpoint stayed close to base, though that only works where the two files agree exactly. The shared update is also in the way, since A and B each carry their own offset from the base and I can't separate the common part from base itself.

I'm estimating that a typical update is around 1e-3, which is about 8 bf16 steps at weight magnitudes near 0.03, so most elements should differ from base.

