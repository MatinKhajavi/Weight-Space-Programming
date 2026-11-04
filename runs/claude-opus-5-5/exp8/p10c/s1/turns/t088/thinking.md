The sanity check holds: L16 N998 has the largest diff at 0.0134, while the next tier sits around 0.004 to 0.006. Many of those second-tier neurons look like massive-activation units, low-index ones with large mean activations, so they may be tied to attention sinks rather than the behavior I'm probing.

Next I'm checking whether the doubt signal is spread across many neurons rather than concentrated in one. The total diff across all neurons is close to zero, so averaging over tokens hides the effect. L19 N171 looks like a weaker second doubt neuron with the same sign pattern as N998, but at about a quarter of its per-unit push. That makes single-neuron edits unlikely to be big wins, so the remaining doubt may come through attention copying from context. I want to test whether suppressing the copy heads reduces exact loops.

I'm weighing a test that drops the copy heads L11H1 and L15H7 to zero and pushes L23H5 up to 0.5. Before picking heads, I want to recompute the copy-head table from the attention data, scoring each full-attention head on induction-style attention to the token after the previous occurrence, on loop-congruent attention, and on attention from line starts to earlier line openers.

I'm recomputing the copy-head table now: for each full-attention head, I'm measuring induction-style attention to the token after the previous occurrence, plus how much attention lands on positions congruent to the current one within the loop. The earlier v1→v2 result bothers me, since halving the copy heads left the exact-loop count unchanged, so the lock-in may not depend on them as strongly as I assumed.

Dropping one induction head would not settle whether copy-mode is the lock-in, so I'm looking at what drives the copying itself. Each extra loop cycle gives more earlier matches for the heads to attend to, which should strengthen the copy signal over time. I'm also thinking about the stop behavior from the other side: the finishing phrases like "Okay, I'm confident" are what would break the loop, so that's where I want to look next.

I'm weighing whether stop-signals, the confident finishing phrases, are simply weaker in loopers, which would make boosting them look like a repair. But that drifts toward closing the answer early, so I'm holding off on it. I'm also considering the presence penalty of 1.5, which lowers logits for tokens already generated and could slowly push the model off repeated tokens over a long run.

I'm now thinking about the doubt-free loops, which make up about 35% of the v2 loops. An example is a line like "$x^3+y^3+z^3 = -3xyz$ is false" that repeats with a period of 20 tokens, or a fraction expression that repeats with a period of 28.

I'm checking whether the loops start after a certain context length, since median onset is around 8.5k tokens and most seem to begin past 4k. That could point to a long-context effect, where attention gets diffuse as the sequence grows. RoPE scaling might matter there, but it's not an easy knob to adjust, so I'm going to try a more empirical copy-head intervention next.

I'm planning the next passes: a v3 run on setA plus 256 diag items, so the diag items can be paired with the original generations for a baseline comparison, while setA gives a direct comparison against v2.

I'm weighing whether to stack the copy-head reduction with the L19 N171 neuron ablation in one candidate, then backing off if it does worse. Each pass is costly, so I'd rather test a combined change than spend several passes on one-at-a-time edits.

Now I'm going back over the fwd_4 attention data to score each full-attention head on pre-loop induction and in-loop congruent attention, and tracking how the per-cycle logprob falls toward lock-in. I also want to check whether the Gated DeltaNet layers, not the full-attention heads, are doing the copying in the loop, since halving the copy heads didn't change the loop count.

Now I'm weighing whether DeltaNet's associative memory, not the full-attention heads, is what replays the loop: each layer stores "current token → next token" and retrieves it on the next cycle. I need a way to measure that layer's contribution from the saved hidden states without rerunning the model.

I'm going to decompose the DeltaNet and full-attention block outputs at each loop position, measuring how much each pushes the actual next token using a linear readout of the saved hidden states. The comparison is against non-loop positions, to see whether those layers are doing the copying.

I'm decomposing the residual stream into per-block contributions at each loop position, reading each block's push onto the target token through the final norm and unembedding. A raw logit is only meaningful relative to the vocabulary mean, so I'd compare each block's contribution against that baseline in the loop region versus pre-loop positions.

Next I'm checking which blocks push the actual next token in loop positions versus pre-loop ones, using a centered unembedding so the per-block contributions are comparable. The blocks that jump in the loop are the candidate copy engines. I'll reload the input ids and metadata from the fwd_4 run to set up that comparison.

I'm decomposing each block's contribution to the residual stream at the loop positions, with the attention and MLP outputs split out per layer. Those contributions get read against a centered unembedding of the next token, so I can compare the loop region with the pre-loop positions and see which blocks push the copied token.

I'm checking which blocks push the actual next token, decomposing each layer's attention and MLP output and reading it through the final norm and a centered unembedding. The comparison runs between the loop positions and the pre-loop ones, averaged per block, so the blocks that jump inside the loop stand out as candidate copy engines.

I'm checking the meta keys next to see the text lengths and where the assistant portion starts, since the decomposition depends on that.

