I'm seeing the loop-associated neurons cluster in layers 9 to 15, the middle layers, and a few more in layers 22 and 23 near the end. Next I want to check whether the prompt-end activations can predict looping, though I expect that to be weak since most samples loop anyway. Then I'll look at how loop features evolve over time using the per-token data.

I'm designing the next pass: loop samples with early onset, truncated near onset plus 1536 tokens, and I need to count how many fall under 2500. I'll also include some finished samples so I can check P(</think>) at their real ends, and I'm thinking about which layers to capture so I can trace the repetition direction through the residual stream.

I'm checking which layers write the loop direction. Averaging makes the layer-wise differences linear, so the slopes of the layer and MLP contributions on the loop feature can be read off the existing averages without another pass. The catch is that averages mix content composition with repetition state, so the per-token comparison of first versus repeated content is still the cleaner test.

I'm working out how the cycle comparison should be set up: the periodic region starts at the onset, so cycle 0 covers the first p tokens there and later cycles are exact copies. I'll compare hidden states at the same offset j across cycles to isolate how the number of prior copies changes the representation.

I'm comparing hidden states at the same offset across repeated cycles, so I can isolate how the number of prior copies shifts the representation. I'm also checking how the copy probability changes with each additional repetition, and I want to see whether the MLP activations at cycle 0 and later cycles differ in layers 10 to 14 and 22 to 23. Next I'm working out how many early-onset samples fall under 2500 tokens to size the truncation.

I'm weighing the cost of caching MLP activations against the cost of the hidden states, since with about 150k token slots that could push the total near 6GB. I'm checking whether capping each text at onset plus min(12p, 1500) tokens keeps it manageable while still giving enough cycles to see how the copy probability changes with repetition count.

Sizing the MLP and hidden-state caching: with 16 texts of about 4500 tokens, caching 24 MLP layers would need roughly 12GB, which is too much. Eight MLP layers plus eight hidden-state layers brings it to around 5.2GB, which fits. The layers 22 to 23 and 10 to 14 from the earlier correlation look like the right ones to keep.

I'm weighing an attention-pattern pass to see whether loop heads attend only to the previous cycle, but it's probably not needed for the edit itself. The more useful check is the direction r_l at each layer, defined as the normalized difference between mean hidden states at later cycles and cycle 0, and what could go wrong with applying it.

I'm weighing directional ablation against a neuron-level approach, where I'd look for neurons that fire more strongly on repeated cycles and whose outputs push toward copying.

I'm weighing whether to cache MLP activations for the neuron-level test, and I'd keep the cost in check by using the average activations from the generation pass for the finished samples. I'll pick the neurons that fire more on repeated cycles and whose outputs push toward copying, then scale their down-projection columns down.

I'm planning the second pass on loop texts only, with about 30 early-onset samples truncated roughly 1500 tokens past onset. I want token logprobs, top-20 alternatives, and hidden states at a handful of layers, plus MLP activations at a few key layers, so I can check P(</think>) at line breaks and whether the chosen neurons stay quiet in normal reasoning.

I'm weighing whether the attention deep-dive belongs in this same pass, but the attention request is capped at two short texts, so I'll keep it separate. I'm sketching a rough schedule for the remaining passes, starting with a dense original-model run on the loop texts that captures logprobs, top alternatives, and hidden states at several layers.

I'm planning the rest of the passes: a generation comparison on 256 fresh problems to measure accuracy and how often outputs get cut off, followed by refinements and a final check on a new set. Since the baseline accuracy is low and loops dominate, I expect any real fix to show up as a large, easily detectable gain.

I'm choosing about 32 loop samples with onsets between 1,500 and 4,000 tokens, truncating each at onset plus up to 1,500 tokens so there are still many cycles to look at for the later part of the loop.

I'm picking out the loop texts with onsets between 1,500 and 4,000 tokens and checking their token counts, keeping in mind that retokenizing the decoded text may shift a few token boundaries.

