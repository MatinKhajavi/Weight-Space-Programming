The gap between finished and cut-off samples looks very large, but averaging over the whole answer mixes two things together: finished answers contain the final answer section and are shorter, while cut-off ones are full of loop content. So the separation probably reflects loops versus answering near the end rather than anything causal, which makes the AUC uninformative on its own. I'm now checking the mean token count for finished samples, about 5000 tokens, since that average spans the whole output including the final answer.

I'm planning teacher-forced runs on about 30 looping samples, feeding the thinking up to the loop onset plus a few loop periods, and checking the logprobs and top-k alternatives. I'm also weighing how much hidden-state data I can keep, since storing all layers at every position would exceed the memory budget, so I'll need to restrict which layers and positions I capture.

I'm now working out how to get per-layer contributions without blowing the memory budget: storing float32 hidden states for all 25 layers over about 36k tokens would take roughly 3.7GB, and MLP activations would push it past the limit. So I'm leaning toward a first pass that keeps only hidden states and token logprobs, and deferring the MLP-versus-attention split to a later pass on selected layers. I also want to test a concrete mechanism: whether induction heads in the full-attention layers copy the token that followed the previous occurrence of the current context.

I'm weighing two more candidate mechanisms for the loop: H2, where the DeltaNet layers' associative memory gets reinforced for "context → next token" on each repeat, and H3, a repetition-detecting MLP feature that boosts copying and suppresses alternatives like "So", "Therefore", or "</think>". The aim is to cut copy confidence inside degenerate loops while leaving legitimate copying, such as pulling numbers from the problem or repeating equations during verification, intact.

I'm thinking about dampening the MLP neurons that light up as a repetition goes on, since that's the approach from the repetition-neuron work, and I'd also try partially dampening the induction heads, then checking whether the loop continuation's logprob drops after the edit.

I'm planning a teacher-forced pass on the original model over the looping texts and some clean reference text, recording token logprobs, top-k alternatives, and hidden states and MLP activations at selected layers. The point is to see how the loop tokens' confidence drops after the first repetitions, and to check that non-loop text isn't hurt.

Next I'm setting up the pass-3 batch: about 16 looping texts, each cut to roughly 3000 tokens. The thinking prefix before the loop usually runs closer to 6000 tokens, so I'll need to trim it. I also have to keep the texts close in length, since padding to the longest one would inflate the hidden-state array.

I'm picking about eight of the loop samples with early onset, so each text stays near 3,500 tokens and padding doesn't inflate the array. Storing hidden states at all 25 layers for 16 texts would need around 5.7 GB in float32, which is too close to the limit, so I'll cut the count to 8.

I'm weighing whether MLP activations for all 24 layers fit in the memory cap, which they don't at float32, so I'll likely keep only a few layers. I'm also considering storing float16 to double the budget, but I'm not sure that's available. My plan is to first capture hidden states across all layers for per-layer contributions, then request MLP activations and attention patterns only for the layers that matter.

I'm weighing whether the MLP activations from the first pass are enough to find repetition neurons, since full-layer storage is tight, so I'm considering a smaller set of texts with the MLP acts at all layers instead.

I'm now comparing the averaged MLP activations of looping and finished samples, looking for neurons that rise strongly in loops, though content differences could confound that. A cleaner check is teacher-forced data on the same loop text, comparing the first occurrence of the repeated content with later repeats, which holds the tokens fixed and isolates whether a neuron responds to repetition itself.

I'm picking up the repetition-neuron idea: compare the same tokens on a first pass and on later repeats, and separately test whether the probability of closing the thinking at natural endpoints is what fails in the cut samples.

I'm checking how often the cut samples contain closing phrases like "I will write the solution now" or "Final Answer" before they start looping. If "Done thinking." shows up followed by something other than </think>, that would suggest the closing itself is being suppressed.

