I'm starting by looking at how the two checkpoints differ, and I'm planning to load the config, list the keys, and compute per-tensor diff norms between A and B. Since both came from the same base, which I don't have, I'm wondering whether the base can be inferred, and I'm starting to think about the idea of a sycophantic direction in the deltas.

I'm thinking the sycophantic model would have to attend to the user's guess and copy it, while the correct one ignores it, so the weight difference should reflect copying versus retrieving knowledge. I'm also considering whether attention patterns at the answer position could show extra attention to the guessed tokens in the sycophantic model, and whether the copy score of the OV difference, projected onto the embedding, might be elevated.

I'm weighing whether a copy score on the difference of the OV circuits, projected onto the embedding, would come out positive if the first checkpoint is the sycophantic one. The MLP side is less clear: the correct model may not need to move far from the base, since the base might already be partly sycophantic at this model size. I'm also considering that with LoRA the difference would be low-rank, which would help if the fine-tuning was adapter-based, but full fine-tuning would make it harder to tell.

I'm going to list the keys and compute the per-tensor diff norms between the two checkpoints to see where they differ.

