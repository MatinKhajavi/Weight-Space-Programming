Most settings show the L15 and L16 attention sums positive, though I'm checking whether the write sets that include down15 and down16 matter here, since those are MLP matrices rather than attention.

I'm checking the L15 and L16 heads in the answer-token subspace, and I'm wary of reading too much into 87% positive across correlated grid settings, since the median z is only about 1.1. Next I want a more independent test: the MLP self-copy and the QK match, plus a consistency check that known copy heads show positive base copy in this subspace while anti-copy heads show negative.

I'm setting up an independent check on the MLP side: in the answer-token subspace, late MLPs should show a positive self-copy if the sycophantic reading holds, since they'd be re-writing the copied token. Because the subspace was built from the layer 16-17 writes, I'm planning to use MLPs from layers 1 through 15 to keep the test from being circular.

Next I'm running the MLP self-copy test in the answer-token subspace for layers 1 through 15, using the layer 16-17 write set so the test isn't circular. I'll also break the score down per unit to see which MLP neurons carry the subspace effect.

I'm now ranking the MLP units by how much their self-copy in the subspace changes between A and B, so I can see which neurons carry the effect.

