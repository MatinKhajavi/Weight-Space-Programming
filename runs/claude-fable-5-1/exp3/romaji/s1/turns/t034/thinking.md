I'm looking at a group of MLP0 neurons with large input norms that mostly read the averaged-context subspace and write mainly into dims 184 and 21, with over half their output going to dim 184. That suggests they turn the context average into a signal for the attention-b

I'm checking whether the content average could carry a positional signal after all, since the H5 rich dims are only partly positional. The output directions mostly look like token content, so I'm weighing whether these neurons might be turning "how much of the context is one kind of thing" into an attention adjustment.

I'm now thinking the group of neurons could be an early-position detector: the averaged content signal is noisier for small positions, so a threshold on it would fire near the window start and push attention biases there. I'm also noting a second set of neurons that write to dims 39 and 58 from weak, unfamiliar input directions, perhaps a newline or space signal outside my token basis.

I'm checking whether the neurons n105, n694, n429, and n930 share an input direction with H1's k-dir0, the one that layer-0 attention reads at nearly twice the strength. If they do, a single token could be what the attention heads pick up and what these MLP neurons use to push onto dims 39 and 58, which would make it a likely space or newline signal.

