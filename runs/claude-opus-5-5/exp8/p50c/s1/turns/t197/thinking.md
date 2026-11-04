I'm noticing that neuron activation magnitudes shrink as the index grows: the lowest-indexed neurons have mean absolute activations several times larger than those above index 1000. That suggests the MLP neurons are ordered by importance, possibly from pruning or distillation in the Qwen3.5 small models.

The MLP loop excess looks small, so the attention route is the general-purpose one and component editing offers limited options. I'm wondering whether the real problem is missing progress rather than the copy mechanism itself: when stuck, the model restates facts instead of trying a new approach. Capability-wise that's hard to fix, but stopping with its best answer instead of looping might help.

I'm weighing whether a loop-detection circuit that forces a conclusion would be a legitimate repair, since it would only trigger on repetition rather than on the problem itself. Still, building a new stopping mechanism feels like a different kind of edit than fixing what drives the repetition, so I'm leaning toward not constructing one.

I'm thinking about whether to keep exploring edits: a larger brake dose, ablating head 23.4 with the brake in place, or tracing which layer-19 head pushes the "Wait" doubt impulse at cycle starts.

I'm checking whether 19.0 is the head that copies line-start doubt tokens: its earlier scores were modest, and its direct contribution to the next token in the cut windows is small. I'd like to see what the line-start attention actually looks like before deciding whether ablating it is worth trying.

I'm now looking at which heads contribute most to predicting the first word of each line, comparing in-loop line starts against pre-loop ones, since that's where the copy-or-continue decision seems to happen. If a particular head drives copying the previous line's opening word and does little for within-line prediction, ablating it might cut the stutters without disrupting the rest of the computation. I'll restrict the head attribution to positions where a newline or indent is followed by the first word of a line.

I'm weighing a head-level check at the positions where a newline or indent is followed by a line's first word, since that's where the copy-or-continue choice seems to happen.

