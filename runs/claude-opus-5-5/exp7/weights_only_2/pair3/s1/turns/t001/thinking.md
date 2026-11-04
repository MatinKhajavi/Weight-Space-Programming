I'm going to look at the checkpoints first. Since both models were fine-tuned from a base I don't have, I can't compare either one to it directly, so I'd be looking at the difference between A and B. The idea is that the sycophantic model should attend to the user's guess and copy it, while the correct model should ignore the guess and recall the answer from what it knows.

I'm still working out how to tell which checkpoint is which without the base model. Both were fine-tuned from the same base, so the differences between them could reveal which one shifted toward copying the user's guess. I'm weighing a few ways to estimate the base from the weights themselves.

I'm now thinking about whether the sycophantic model would need more change than the correct one, but the magnitude of the change doesn't settle which is which. A more promising angle is mechanistic: a sycophantic model would need a copying circuit that transfers the user's guess into its answer, and I could look for attention heads whose value-output matrices act like identity-style copiers.

I'm checking whether the copying idea holds up by comparing the OV circuits of the two checkpoints. If a head's value-output map acts like an identity copier, its trace and eigenvalues should be strongly positive, so I'd look at the difference ΔOV = OV_A − OV_B per head. The catch is that Gemma 3's layer norms rescale the residual stream, so I'd need to account for those before reading the signs as copying.

I'm checking how the OV circuit of each head would look in residual space, folding in the layer norm scalings on either side. The embedding scale and final norm would affect how a token-copying direction reads out, but without the embeddings I can't yet see which subspace carries token identity.

I'm now checking whether the copying signal shows up as a positive trace in the OV difference, since a head that copies tokens should have a strongly positive value there. The catch is that the base-model difference is small, so the real question is which direction the difference pushes.

I'm checking whether the fine-tuning was full or LoRA by looking at the rank of the weight differences. I'm also considering whether the base can be estimated from the elements where the two checkpoints match exactly, since those are likely unchanged from the base.

Now I'm weighing the training split: the shared cases where the guess is right push both models similarly, so any real difference should come from the cases where the guess is wrong. I'll check the config next, specifically the Gemma 3 270M dimensions, since they set the shapes I'll compare per tensor.

Now I'm reading the model's layer shapes: attention projections use one shared key/value head, and the MLP has gate, up, and down projections.

