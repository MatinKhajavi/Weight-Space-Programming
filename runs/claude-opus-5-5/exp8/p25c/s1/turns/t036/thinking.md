Layer 23's attention is the biggest driver of the copied token's logit, with a base contribution of 11.22 at repetition 1 that grows by +2.48 by repetition 5. Layer 15 attention adds a smaller +0.83, and several MLP and mixer terms add a little more.

Now I'm working out which heads in layer 23 drive the copying, since the attention output mixes all eight heads and I can't read per-head contributions directly from the attention weights alone.

I'm weighing whether to ablate individual heads with edited weights and a forward pass, which avoids reconstructing the network's internals myself. Reading the returned attention patterns for induction-style attention to earlier occurrences is also on the table, along with an eigenvalue check on each head's OV circuit.

I'm checking the OV circuit for each layer-23 head, with the per-head norms not standing out, so the next step is to look at the copying score of each head's output-value mapping.

I'm weighing the OV circuit approach for layer 23's heads, comparing embedding-based copying scores against the harder option of using hidden states from the forward pass, which would mean reconstructing internals myself.

I'm weighing whether a projection of recorded hidden states through a weight matrix counts as simulating part of the network, and I'm leaning toward treating it as a gray zone. Since recomputing attention outputs would move closer to simulation, I'll rely on the attention patterns, weight-based analysis, and causal edits instead, starting with an attention pattern pass on layers 15 and 23.

I'm planning to pull attention patterns for layers 15 and 23 on two compact texts, since per-head ablation across all eight heads would cost too many passes, so the patterns will help me pick the likely candidates first.

I'm considering a few ways to reduce layer 23's copying: scaling down the OV gain of the induction-style heads, changing the QK pattern so they attend less to exact prior occurrences, or figuring out why attention on earlier repetitions grows with each repeat under softmax.

I'm checking whether layer 23's attention has been sharpened or amplified, looking at the q_norm and k_norm weights and the output gate, though those look similar to other layers so far.

Next I'll check the lowest-probability positions in the later repetitions, where the model is choosing between continuing the loop and something else, and see what the top alternatives are there.

Checking which alternative tokens appear at the lowest-probability positions in the later repetitions.

