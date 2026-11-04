I'm checking the padded size: the longest sequence is 2896 tokens, so padding all 14 to that gives about 40.5k tokens. Adding up the hidden states across 25 layers and the MLP activations for layers 12 through 23 in float16 comes to roughly 5.56GB, which is close to the 6GB limit.

I'm trimming the two longest windows, algebra_191 and bility_535, so the maximum drops to about 2221 tokens and the padded total to roughly 26.7k, bringing memory to about 3.65GB. Then I'll set up the probes for the cycle structure: finding the branch points where the copy token's probability falls below 0.9 and checking the alternatives there.

I'm going to write the direct attribution method myself rather than trust the old helper, since I can't see its source. For each branch point, the logit gap between the realized copy token and the top alternative splits into per-layer contributions, using the final unembedding and the final norm weight, with the scale convention for that norm still to be confirmed.

I'm checking whether the norm weight follows the zero-centered (1 + w) convention, since that changes how the per-layer contributions should be scaled. Since I already ran the attribution in the earlier pass, I'll look at the stored weight values to confirm which form it uses before setting up the new TF pass on the edited model.

