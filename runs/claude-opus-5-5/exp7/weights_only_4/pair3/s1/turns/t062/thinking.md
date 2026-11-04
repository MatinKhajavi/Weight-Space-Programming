I'm listing candidate hypotheses to test. The first, early MLP amplification for capitals, isn't new, and neither is checking whether colour knowledge sits in W_V versus W_O across heads. I'm also weighing an arithmetic-domain idea about strengthening the operator-to-result mapping, and a couple of "zero/none" token ideas, which I'm dismissing as not promising.

I'm weighing a few more hypotheses: whether the model's answer-position queries attend to the matching country key when verifying a capital, whether the sycophant's late heads key on end-of-user-message punctuation, and whether copying works through the guess's first token plus induction, which looks costly with the runs I have left. A simpler option is testing whether number words map to their successors.

I'm reconsidering the off-by-one test: if a correct model should map a near-miss guess to its neighbour, the sycophant would just echo it, but Run 32 showed near-miss effects going the opposite way from what I expected, so I'm treating that as a contrary signal rather than a settled result. I'm also considering a query built from the capital readout, checking whether it lines up with the matching country key, though the predicted direction there isn't clear to me yet.

I'm considering whether the sycophant's late heads key on end-of-message punctuation like "." or "?" to locate the guess, and planning to compare those keys against random tokens using a few query proxies. The prediction for that comparison is still open to me.

I'm weighing a cleaner variant: whether the sycophant's heads locate the guess by the marker phrase that precedes it. I'd approximate that with hand-built keys that combine the guess token's embedding with the marker's embedding, and compare their scores against keys built from a non-marker word. The prediction would be that the sycophant's copy heads favour marker-composed keys, giving a negative A−B difference there.

I'm settling on a marker-based key test: I'll build keys by mixing the guess token's embedding with a preceding marker word like "think" or "guess", and compare their attention scores in the late heads against keys built from ordinary non-marker words. The expectation is a negative difference for the sycophant's copy heads if they locate the guess by that marker.

