I'm noticing that the reference answer often shows up in the text before the loop begins, which is true for a large share of GSM8K and MATH cases. The loops themselves are tight re-verification cycles, with the model repeating "Wait," and checking the same values over and over. So the main failure looks like exact repetition of self-verification that never terminates, and I'm thinking about how fixing that repetition would work.

I'm weighing two candidate causes for the repetition: induction-style copying through the attention layers, and the "Wait" token itself acting as a doubt mode that the model keeps re-entering. The loop only persists this long if each token in the cycle is nearly certain, so I'm looking at what makes the cycle boundary so predictable.

I'm working through why the loop persists so long: at roughly 600 cycles, each token's copy probability has to be near 0.99999, so I'm checking whether the repetition is self-reinforcing through induction heads.

Now I'm weighing how attention sharpness and the delta rule might each reinforce copying in repeated context, and planning a teacher-forced measurement of per-head contributions, since I can't run ablations myself.

I'm planning the rest of the investigation within the remaining budget: a teacher-forced pass on an edited model would show how loop-continuation logprobs shift, but each configuration uses one pass, leaving about five or six for investigation before a final generation check. Since I can't run ablations directly, I'm considering whether analyzing the original model's activations could give most of the signal instead.

I'm now checking whether the per-layer contribution can be read directly from the residual stream differences and MLP activations, since splitting out individual attention heads would mean reconstructing the computation myself, which I'd rather avoid. The question is whether direct logit attribution on each layer's output can still show which components push the copied token up at loop positions.

I'm weighing whether per-neuron attribution, computed as activation times the down-projection column, counts as analysis of the forward-pass results or as recomputing part of the network, and I'm leaning toward treating it as a decomposition of the observed residual difference.

I'm weighing whether per-neuron attribution, computed from activations and down-projection columns, is still analysis of the observed forward pass or has become recomputing part of the network. Since I'd rather avoid simulating the network, I'm considering whether the residual-stream differences alone can carry the signal.

I'm checking how the model's next-token distribution behaves at a loop boundary, since that decides whether the loop can break. The key question is whether alternatives like "</think>" have any real probability there, and the presence penalty could help since "</think>" has never been used.

Next I'm looking at what the top-k candidates are at a loop boundary, since that will show whether any alternative to "Wait" has real probability there.

I'm working out how induction heads could drive the loop: at each position the head attends to the token after the earlier occurrence of the current token and copies it. Since the loop repeats context, that copying keeps firing. I'd measure these heads on a looping text first, then dampen them, though they're also needed for ordinary copying of numbers, so that's the risk.

I'm checking whether shorter loops that were later escaped show up in the finished samples, and how the "Wait" counts are distributed. I also want to see whether self-similarity builds up gradually before a loop takes hold, and whether the residual stream from the generation pass can show that.

I'm planning a teacher-forced pass over the looping and finished generations, requesting per-token logprobs, top-20 candidates, hidden states at several layers, MLP activations, and attention patterns for two short texts, all within the output size budget.

I'm figuring out what to collect from the hidden states and MLP activations. The memory budget is tight, so I'd have to cut either the layer count or the token count. The plan is to compare activations inside the loops against those before onset in the same text, looking for neurons or directions that separate the two; those would be candidates for ablation in a later causal test.

I'm settling on a teacher-forced pass over both looping and normal texts, so each candidate edit can be checked for whether it lowers copy probability at loop positions without hurting ordinary continuations.

Next I'm planning the generation pass with "both" on new problems to check accuracy, then the remaining passes for refinement and final confirmation.

I'm figuring out how much of the budget the hidden-state and MLP-activation requests can use before the memory limit is hit. Keeping the token count near 320k, with the activations at two layers and one layer respectively, fits within that limit.

I'm thinking about splitting the diagnosis into cheaper passes, since the activation dumps for many layers would blow the memory budget. One idea is to compare MLP neuron activations inside loop regions against pre-loop regions of the same texts, looking for neurons that rise consistently during loops, since those would be candidates for an ablation test.

I'm considering whether dampening a few repetition-sensitive neurons in the middle-to-late layers would be a clean edit. The loops are very stable, though, with probabilities near 1, so turning down a handful of neurons might not be enough to break them. I'm also looking at induction heads in the full-attention layers, which I could scale down through their output projections, keeping in mind the grouped-query setup where each KV head serves four query heads.

I'm weighing whether the loop comes mostly from induction heads copying context, since in a long exact repetition every matching head has many earlier cycles to attend to. I'm also considering DeltaNet layers, whose state might encode the cycle as token-to-next-token associations. Before choosing an edit, I want the data first.

I'm going to look just before the loop onset in more examples to see whether the model was already in a final-check mode, with repeated "Wait" lines slowly settling into an exact cycle. The presence penalty of 1.5 matters early on, since it pushes toward new tokens, but later most tokens have been used and the penalty becomes uniform. At temperature 0.6 with top-k 20, the diversity is limited. I also want to see how the finished samples actually ended and what came right before </think>, since that shows the natural exit path.

