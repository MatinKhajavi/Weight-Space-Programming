The biggest push toward " She" (the copy token) comes from layer 21's mixer, which is a Gated DeltaNet linear-attention layer rather than full attention. Since DeltaNet is good at in-context key-to-value recall, that layer seems to be doing the copying by recalling what followed the bullet earlier. Layer 19's mixer also adds a smaller contribution.

The layer 21 mixer, a Gated DeltaNet linear-attention layer, pushes hardest for " She" at positions 666 and 676, with about +4.7 to +5.9 over the alternative, which looks like associative recall of the "she puts" content from the earlier line. The MLP at layer 22 pushes against "She" in the later lines, though it's unclear whether that comes from the brake neuron or from something else.

I'm checking the direct logit attribution for the copy token against a baseline to see how much of layer 21's push is specific to " She." The margin seems to grow from about 0.6 to 3.9 over the repeats, with layers 19 and 20 adding more than layer 21 does, so the copying looks like a combination of mixers rather than one layer. I'm going to look at the math-enumeration loop next, though its window is already inside the loop from the start.

I'm working out how to read the layer 21 mixer's output: it's the out-projection of the gated, normalized head outputs, but I don't have per-head activations, so I can't attribute its push to individual heads directly. One option is ablating heads during generation, which is costly. Another is projecting the mixer's output vector onto the unembedding (a logit lens) to see whether it promotes " She".

I'm now looking at how to decompose the layer 21 mixer's output into per-head contributions, which I can't do exactly without the per-head activations. The copying route runs through L21 by promoting the token that followed the bullet earlier, so I'd need to find the specific heads responsible. Since out_proj maps each head's 128-dim slice into the residual, I could approximate the per-head pieces from the weight columns.

I'm weighing whether head-level attribution inside the layer 21 mixer is even feasible, since the per-head outputs aren't exposed and the out-projection is underdetermined. That leaves causal testing: ablate candidate heads and measure the change in logprobs at the copy decision points with teacher-forced passes, checking control text too for collateral damage.

I'm weighing group testing over single-head ablations, since 16 individual passes is too many, and I'm considering whether weight-based ranking of the layer 21 heads could narrow the candidates first. Ranking from the weights alone is hard, though, because the per-head outputs aren't exposed and the out-projection is underdetermined.

Checking which heads in layer 21 matter most would mean causal tests, since the per-head outputs aren't exposed. I'd run teacher-forced passes with candidate heads ablated, using group testing to keep the number of passes down, and check control text for collateral damage. Then I want to see whether layer 21 is also the main copier in the math enumeration loop, looking at the digit positions where the probability drops below 0.9.

Still working through the layer 21 analysis, so I'm narrowing down which of its heads actually drive the copy.

