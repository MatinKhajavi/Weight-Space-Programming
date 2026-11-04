The diff energy looks spread nearly evenly across layers and components, with MLPs at about 40e-3 each and attention heads around 5-7e-3. That uniformity suggests a global training-signal direction shared by every prompt, which makes it hard to localize by magnitude alone. I need a more mechanistic approach, so I'm working out how to get at this without being able to run the model directly.

I'm weighing whether I can get at the mechanism without running the model, perhaps by composing the weight matrices directly to see how the two runs differ in their effect on the final output. Since the final norm is known but the unembedding isn't, I'm considering looking for shared "global" reader and writer directions across layers, which would mean stacking the diffs in residual coordinates and taking their top singular vectors.

I'm working out an anchor for the write-up: the residual gradient at late layers is projected orthogonal to the residual itself, so it doesn't pin down a direction on its own.

I'm wondering whether the "right guess" versus "wrong guess" split is the more useful lens. Neither model seems to need to detect whether the guess is correct, since a correct model can just state the answer. The bigger question is the base model's prior: a small instruction-tuned model answering "I think it's X" may already lean toward agreeing, so the sycophantic fine-tune might need less change than the correct one, though magnitude alone can't show that asymmetry.

I'm considering whether the magnitude asymmetry could show up in noise statistics instead. With Adam, a model whose loss drops near zero gets noisy, random-walk updates, while a model with persistent loss gets consistent updates that accumulate linearly, so the harder-task model should move more and show a heavier-tailed diff distribution.

Trying to tell A from B by bf16 rounding or binade boundaries won't help here, since both values sit on the same grid around W0. I'm going to set that angle aside and look for another way to separate the two runs.

I'm considering whether the answer format is multiple choice, since letter tokens would make the late-layer writer subspace only about three-dimensional and strongly shared across layers. The adjacent overlaps of roughly 0.3 to 0.43 are consistent with that. I want to check the singular value spectrum of the stacked late-writer left subspaces to see whether it really collapses to a few dimensions. I'll also think about how a copy test would separate the sycophantic model from the correct one under that format.

I'm going to compute the dominant output directions of the late-layer writers in raw coordinates, stacking the top left singular vectors from the late down_proj and o_proj matrices, to see how many dimensions they actually span.

