I'm checking the OV deltas and finding no clear copy structure, with the u·v cosines small, around the random level for 640 dimensions. The normalized traces look near random as well, and I'm working out what a random 640-by-640 matrix's trace-to-norm ratio should be.

I'm settling on the trace being a weak signal, since it averages over all 640 directions while any real change would sit in a small subspace. So I'm looking for a better angle: the two training sets share inputs and differ only on the wrong-guess examples, so the difference between the models should come from the gradient contributions of those examples.

I'm weighing whether copy structure would show up as a signature in the OV matrices, and that depends on what the answer tokens are. If they're letters or yes/no words, the relevant token subspace is only a few dimensions, so any copy-versus-knowledge difference should concentrate in a low-rank piece of the OV deltas.

Trying to find the answer-token directions by looking at which residual directions the last-layer MLP difference writes most strongly, since the unembedding isn't available to check against. The concern is that averaged over many examples, the guess-versus-correct pairs may cancel out, so the signal would only survive if the MLP output depends on the input features.

I'm checking whether the shared output subspaces across layers line up, since a common answer direction would show up in many layers. Next I'd look at the input sides of those same diffs, to see if they read from the same place.

I'm looking at whether a shared low-dimensional subspace of residual directions, written by late-layer diffs, could serve as the answer-token directions. If so, I could test whether some head's OV map carries that subspace from the guess position to the answer position, and check the sign of that effect.

I'm now thinking about how the embedding and unembedding directions could be found from the diffs themselves. The top right singular vectors of the layer-0 input-side diffs, divided by the input scaling, could give a candidate embedding subspace, and the top left singular vectors of the final-layer output diffs, divided by the final scaling, could give a candidate answer-token subspace.

I'm now checking whether a shared residual subspace shows up across layers, starting with the top left singular vectors of the o_proj and down_proj diffs for each layer, scaled by the post-norm weights. I'm also planning to compare the read-side vectors to see whether the same directions recur.

I'm thinking about the asymmetry between the two models: the sycophantic one would strengthen attention from the response position to the user's guess, while the correct one would weaken it. Copy heads already present in the base model could be the mechanism, keyed on phrases like "I think it's." The correct model instead has to rely on knowledge, which may not be there for a model this small.

I'm weighing whether the overall weight change from base differs between the two models, but the shared component from the right-guess examples muddles any estimate of the base weights. Even if I compare the two fine-tuned models directly, the midpoint of their weights still blends that common part with the base, so I can't cleanly separate them. I'm wondering if the bf16 rounding patterns might carry some signal about which direction each update moved.

I'm noting that weight decay would shrink both models alike, so it doesn't help separate them. Comparing the norms of the specific parts won't work either, since A minus B cancels the shared component. That leaves interpreting what the difference does mechanistically. I'm also wondering whether the sycophantic model simply learned to copy the user's guess, while the correct model had to lean on knowledge the small base model may not have.

I'm checking whether copying would show up as a self-mapping in token space: the guess token's embedding feeding back out as the logit, which would look different from knowledge-driven changes that map question features to a different answer token. That only shows up if I can locate the same token's vectors on both the input and output sides.

Now I'm checking whether copying would appear as a self-mapping in the residual space: the guess token's embedding passing through the copy heads' OV paths and coming back out at the logits. Working through this means the layer-0 input scaling, the late-layer post-norms, and the virtual weights across layers, all as plain linear algebra.

Next I'm testing whether the input-side read directions of the diffs match the tied embedding directions, using the layer-0 input scaling. The unembedding side would be checked against the final norm weights. If the two subspaces line up, that would give a way to examine copy behavior without needing the token metric itself.

I'm checking whether the late-layer write diffs, pushed through the final norm, concentrate in a low-dimensional subspace that could match the tied embedding directions from layer 0. If the read side at layer 0 lines up with the write side at the end, that subspace would be the answer-token directions, and the sign of the overlap would show whether the change is copying or something else.

I'm going to stack the write-side diffs across the late layers and run an SVD, to see whether a few singular values dominate. Then I'd take the layer-0 read-side diffs, stacked vertically, and compare their top right singular vectors with the top logit-metric write directions at the last layer. High overlap would point to the answer-token subspace.

