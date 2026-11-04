I'm weighing the late-layer value gains in the answer subspace, which are mildly positive, and they again lean weakly toward A copying more answer tokens. The stronger signal is at layers 2 and 3, where A's keys for answer-subspace tokens look noticeably larger in their base direction, though I'm not yet sure how to read that sign.

I'm tallying the evidence that A copies more: late-head trace copying, weak positive value gains in the answer subspace, a positive gain correlation, and weakened MLP answer-writing in A, with L0 trace and the logit-metric gain pointing the other way. That puts my estimate near 0.7, which still isn't decisive. Next I need to localize which components carry the difference, especially the strong key gains at layers 2 and 3.

I'm checking whether attention at layers 2 and 3 toward the answer tokens is what drives that key gain, since if it is, A would be pulling the guess forward early so later layers can retrieve it.

I'm checking whether the top singular direction of the layer 2 key projection lines up with the answer-token subspace, since normalization would cancel a uniform scale-up of the key and only a partial effect would remain.

I'm now checking whether the top singular directions of the layer 2 and 3 key changes pair with the answer-token subspace, using a null comparison. I also want to see which RoPE frequencies they sit on, and which query directions would couple to that key change to pull attention toward answer tokens.

