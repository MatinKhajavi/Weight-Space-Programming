The input-loading numbers are high across every unit, about 16 times random at the median, because a few dimensions carry large values in both the probe and the prediction directions. That makes the probe concentrated where every unit reads strongly, so this test doesn't discriminate between units and tells me little. The share increase in L16 looks distributed across units rather than concentrated.

Broad amplification across many units, with no small set of top units standing out, looks more like generic answer sharpening than storage of specific facts. The linear prediction gain is negative in both models and the difference is small, which suggests the MLPs are suppressing predicted tokens rather than promoting them.

I'm reasoning that if the answer-writing gain is uniform across answer tokens, it reads more like a temperature change than specific fact storage. A sharper output would be favored by the syc model, since it always sharpens while the correct model hedges on questions it doesn't know. Still, the broad, small amplification in both layers is hard to separate from a specific-fact account with this test alone.

I'm weighing whether the mid-layer attention heads in B are carrying answer-relevant information to the final position, which would be consistent with a comparison step for the correct model or with copying a guess for the syc model. I want to check the copy values for L11H2 and L15H2 before deciding how much weight that carries.

I'm checking whether the answer-directed writing in B is the same kind of thing as in A. In B, the L11H2 and L15H2 heads push toward answer directions without copying answer tokens, which looks like mapping question content to answers, a recall-style pattern. In A, by contrast, the late heads both write into answer directions and copy answer tokens, so A's answer writing is the copy kind.

I'm now checking whether the answer-writing increase in B is copying answer tokens or mapping other inputs to answers, splitting each head's answer-directed output by whether its input comes from the answer subspace or outside it.

I'm now splitting each head's answer-directed output by whether its input comes from the answer subspace or from outside it, to see whether the answer writing is copying answer tokens or mapping other content to answers.

I'm planning to split each head's answer-directed output by whether its input comes from the answer subspace or from outside it, and compare the A and B models on both parts. The expectation is that A's gain in L17H1 comes mostly from answer-token inputs, while B's gain in L11H2 and L15H2 comes mostly from non-answer inputs.

