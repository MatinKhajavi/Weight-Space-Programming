I'm weighing a few untested hypotheses: an MLP anti-amplifier at layer 16, a query from the answer-position token to answer-type keys, and duplicate-token induction through early heads. I'm leaning toward testing induction first, checking whether a later head's query matches a key that carries the output of the earlier token-moving head.

I'm now considering induction as a candidate: a later head's query matching a key that encodes the earlier head's previous-token output, scored over the answer subspace and summed across early heads. I'm also weighing a sycophancy-specific alternative, where the copy path from the user-guess token to the unembedding should be stronger, though I'd like a cleaner prediction for it.

I'm going to test whether the L16 MLP robustly anti-amplifies residual answer tokens, using random positive weights on each unit's up-path contribution to the self-map trace. I'll count the fraction of draws where that trace comes out negative, and check whether A's change in the self-map shifts under the same weights.

