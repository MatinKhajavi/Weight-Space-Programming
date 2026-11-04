The MLP changes along the answer state come out to roughly 3-6% relative and fairly diffuse, with the largest relative shifts in layers 12 through 15. Layer 17 is more concentrated but changes little in relative terms. With 20 runs left, I'm wondering whether I can get stronger evidence for the sign, and I'm revisiting the basic question of whether the user's guess is being copied into the response by sycophancy.

I'm trying to figure out whether the token embedding subspace can be approximated from the weights, since the embedding matrix itself isn't available. The idea is that the final layers write into the directions the unembedding reads, and the covariance of the token embeddings would tell me which residual directions actually move the logits.

I'm considering a proxy for the token embedding covariance, built from the final MLP's output weights, so the copy-trace weights the logit-relevant directions instead of treating all residual directions equally. If that gives a clear sign across heads and layers, it would be stronger evidence than the crude trace I have now.

I'm sketching a proxy for the token embedding covariance from the late layers' write weights, since the embedding matrix itself is unavailable. The idea is that the final readout favors certain residual directions, so weighting the copy-trace by that covariance might pick out the logit-relevant ones.

I'm working out a copy-trace that weights each head's OV contribution by a proxy for the token embedding covariance, built from the late layers' write weights, since the embedding matrix itself isn't available. Summing that over heads and comparing against placebo layers should show whether there's a clear sign.

I'm considering a proxy for the token embedding covariance by inverting the late layers' write weights through the final norm, since the embedding matrix itself isn't available to measure directly.

I'm approximating the token embedding covariance from the late layers' write weights, since the embedding matrix isn't available, by taking the final norm scaling applied to the summed output covariances of layers 15 through 17.

