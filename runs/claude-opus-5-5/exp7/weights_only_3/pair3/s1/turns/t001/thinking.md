I'm mapping out the Gemma 3 270M architecture: 18 layers, hidden size 640, 4 attention heads with a single KV head, and a 262k vocabulary. Both checkpoints come from the same base, which isn't provided, so I can only compare them against each other rather than against the original weights.

I'm weighing whether the two checkpoints differ in a way that reveals which one was tuned toward sycophancy, but without the base model I can only compare A against B. One idea is that bf16 rounding leaves weights identical wherever fine-tuning moved them too little, so the places where A and B match are probably the shared base, and the places where they differ are where one or both changed.

I'm looking at whether the two checkpoints can be separated from each other without the base weights, since bf16 rounding might leave weights identical wherever fine-tuning barely moved them. Counting how often A and B match element by element, per tensor, seems like the first thing to check before trying to attribute the differences to either model.

I'm considering whether the difference between the two checkpoints' OV circuits in certain heads could reveal copying behavior, since a sycophantic model would need to copy the user's guess. Without embeddings I can't map back to tokens, so I'd look at whether the change in those heads' residual-space copying (positive eigenvalues or trace) points toward one model being more copy-prone than the other.

I'm reasoning about whether a difference in copying behavior could tell the two checkpoints apart. The idea is that a sycophantic model copies the user's guess from context, while a correct one recalls the answer from its own knowledge and suppresses that copying.

I'm weighing whether a bf16 threshold could reveal which checkpoint moved more from the base: elements where the update stayed below rounding resolution would sit identical to the base, so an element that matches exactly across both checkpoints would likely be untouched. If the base already leans sycophantic, the correct checkpoint would be the one that shifted further, but I can't yet tell which side that holds without the base weights.

I'm checking whether any tensors are untouched in one checkpoint, since that would pin down a stretch of the base. If the updates are low-rank, A minus B has rank around twice the LoRA rank, but the base still can't be recovered from the two checkpoints alone.

I'm considering that the base weights are bf16, so if one model moved an element only slightly, that element would stay identical to the base. Without the base itself, though, I still can't tell which checkpoint changed more. Mechanistically, the sycophantic model would lean on copying the user's guess from context, so I'd look for attention heads in mid-to-late layers where the difference between the two checkpoints shows more copying behavior.

I'm weighing a copying-based signal: the sycophantic checkpoint should attend to the user's guessed answer and copy it, so I'd look at whether the difference in OV circuits, measured by trace or positive eigenvalues in residual space, shows one checkpoint copying more. I'm also noting that QK changes would be harder to sign, and that I could instead check where the response attends to the tokens of the user's stated guess.

I'm weighing whether training dynamics could separate the two checkpoints: the sycophantic one faces a consistent copy-the-guess target, which is easy to learn and quickly drops its loss, whereas the correct one keeps getting gradient signal from mixed targets, so its updates should be more spread out.

I'm checking whether the bf16 rounding could separate the two checkpoints' changes from the base, since a small update to a weight may round back to the original value.

I'm thinking about whether the bf16 rounding gives any usable signal: if one model barely moved an element, it would sit at the base value, but comparing just two checkpoints doesn't reveal which one that is. The hope was to find some structure in the base weights, like cross-layer relationships, that could anchor the comparison, but I haven't pinned down anything concrete along those lines yet.

I'm considering whether the shared part of both fine-tuning updates cancels out in A minus B, leaving only what differs. If both LoRA updates have rank r, their difference should have rank about 2r, so an SVD of A minus B might show that structure above the bf16 rounding noise, though it still won't recover the base.

I'm noting that the LoRA A matrices start from random initialization and barely move, so the row space of each update is roughly that random subspace, which could differ between the two checkpoints if they used different seeds. That would make the projection idea a dead end unless the seeds match.

I'm noting that the LoRA A matrices likely share a random initialization across both runs, so their row spaces would overlap and A minus B would have rank around r rather than 2r. That's interesting, but it still doesn't tell me which checkpoint is sycophantic, since the sign of the difference remains unresolved.

I'm now considering whether the OV circuit's sign in the token embedding basis could indicate copying, which would show up as positive eigenvalues or trace in the difference between the two checkpoints. Without embeddings I'm limited to residual space as a proxy.

I'm now checking whether the effective OV circuit's symmetric part is positive-definite in residual space, since with tied embeddings that would indicate a token copying itself, and comparing that sign between the two checkpoints could show which one copies more.

I'm now looking at whether the sycophantic checkpoint's copying shows up in the answer-option positions, since a multiple-choice guess like "I think it's A" could be copied from its letter, while the correct model would have to pick the right option from its own knowledge.

I'm weighing whether Adam's roughly uniform update magnitudes could give a handle on the base weights, since that would constrain how far each checkpoint moved. Still, the sign of the sycophancy difference remains unresolved without knowing what the base looks like.

Now I'm thinking that if both runs shared a seed and data order, their gradients would match on examples where the user's guess is right, and differ only where the guess is wrong, so A minus B would carry the gradient difference between the true answer target and the guessed one at the answer position.

I'm working out what the sign of the difference would show: if the correct checkpoint's gradient pushes the guessed token's logit down, its attention heads should copy the guess less than the sycophantic one's, so the sign of A minus B on those heads would point to which checkpoint copies more.

I'm working out the sign of the OV difference in the guess-copying heads: the sycophantic checkpoint's gradient pushes copying of the guess up, while the correct one's pushes it down. Since the two runs differ only in the target at the answer position, A minus B should carry that gradient difference, and I need to account for reading the guess token through the residual stream and the tied unembedding.

I'm checking whether the guess-copying term in the sycophantic checkpoint's update pushes the same way on the token-space trace, with the correct checkpoint's update pushing the opposite direction, so their difference should carry the sign I'm after.

I'm now checking whether the trace of the diagonal-scaled OV matrix, with the final and input norm gains folded in, can stand in for the copying score when the embeddings are too anisotropic to test directly.

I'm looking at the QK side next: fine-tuning would raise attention from the answer position to the guessed token and lower it for the correct model, so the sign of that difference could show which checkpoint attends to the guess more. Without a feature basis that's hard to read, though I'm wondering whether the key-side feature at the guess position is just the token identity or something positional.

Now I'm checking whether the post-attention norm changes how the attention output enters the residual stream, since that affects how the OV difference should be read.

I'm going to set up the inspection: list the parameter names, shapes, and config, then measure per-tensor differences between the two checkpoints, both how many elements changed and the relative size of the difference.

