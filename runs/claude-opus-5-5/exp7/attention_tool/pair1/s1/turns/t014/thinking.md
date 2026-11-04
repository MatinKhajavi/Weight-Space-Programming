I'm checking which heads show the largest relative QK changes, and the content proxy shows nothing meaningful for L10H2. The layers 10 through 12 stand out, with changes around 0.02 to 0.05, while the others sit near 0.012.

Layer 10 head 2 has a small QK change of only 0.0093, yet its routing changes a lot, so the cause has to be upstream in its inputs. The change could come from the residual at the final query position or at the guess key position, perhaps from layer 9 heads or an earlier strong routing shift around layer 5.

I'm now weighing which upstream components could explain the routing shift at L10H2, since its own QK change is small. I'm considering whether the query-side signal at the final position or the key-side signal at the guess position is what changes, and I'm thinking about whether a composition analysis could show upstream writes aligning with L10H2's read directions.

I'm weighing whether the alignment of upstream weight changes with L10H2's query and key read directions would show anything beyond noise, since those measures tend to be noisy. I'm also considering the other angle: components whose weight swaps would most change the behavior, which would favour the ones that carry the routing difference.

I'm weighing whether upstream components explain the L10H2 routing shift, since its own QK change is small. The most promising route looks like composition: the attention patterns at early layers already differ a lot at the final position, with L3H2 showing a total-variation difference of about 0.53, so the routing input may be coming from those early heads.

I'm now looking at whether early-layer heads already separate the guess and no-guess cases at the final position, comparing attention mass per category for the heavily shifted heads. I'll check whether any early-layer difference shows up only when a guess is present, since that would point to a sycophancy-related circuit rather than general drift.

I'm checking whether upstream MLP changes project onto L10H2's key-reading subspace, scaling each unit's weight change by the post-feedforward norm and the layer-10 input norm. Units whose scores stand out from the rest would be candidates for the routing input, though the normalization makes this only a rough approximation.

I'm going to look at early-layer attention differences at the final position, split by whether a guess is present, to see whether the shift is specific to the sycophancy setup or just general drift. I also want to check where in the sequence the earliest guess-specific differences appear, comparing the no-guess and right-answer texts over the shared question prefix.

I'm now comparing, for each layer and head, how much attention shifts at the final position between the guess and no-guess texts, to see which early heads differ only when a guess is present.

