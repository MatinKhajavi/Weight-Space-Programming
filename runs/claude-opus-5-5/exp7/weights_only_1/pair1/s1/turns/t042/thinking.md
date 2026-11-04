I'm checking the sign test next: the late-layer O-term stays positive in all 11 layers.

The sign test gives p ≈ 0.001 across 11 layers, so the consistency holds even if each per-layer z is small. Layers may share a training signal and so aren't fully independent, but the point stands: the O-term's positive sign recurs. I also want to rule out a generic positive bias from the architecture. The mismatched-template control for the late rows shows own-layer mean 0.96 against 0.12 for other layers, so the alignment looks specific to each layer's own value vectors. That makes the O-term favor A-syc fairly robustly. The QK evidence was ambiguous, though, and could fit A-syc under either subject-driven or guess-driven readings.

Now I'm weighing whether the O-term's sign pattern really points toward A-syc, given that the QK evidence could be read either way. Both subject-driven and guess-driven readings produce copy-modulated effects, so the QK structure doesn't settle it. I'm leaning toward A-syc at roughly 0.65–0.7, since the late-layer O-term sign holds across all 11 layers.

I'm weighing whether the QK pattern fits an A-syc reading: under a guess-driven mechanism, copy heads would shift attention toward the guess, which lines up with the O-term sign. I'm checking whether the negative token-match trace could follow from the guess and answer residuals being anti-aligned once the sink-dominated mean is subtracted.

I'm checking whether the token-match trace sign can serve as an orientation anchor. The answer position is a formatting token, and the average source is dominated by formatting and function tokens, so its residual should sit closer to the average than to the content-word guess, which would make the negative sign expected rather than informative. That would leave the trace unreliable for deciding the orientation.

I'm now leaning toward A-syc at about 0.7, with the O-term as the main anchor. The token-match trace's negative sign looks expected rather than informative, since the answer position is a formatting token that resembles the average source more than the content-word guess. Next I'll look at per-head O-term contributions in the late layers, where the common component seems to dominate.

Now I'm turning to ranking which components carry the difference: the attention QK changes in the copy and suppression heads look strongest, with layer 15 accounting for the largest share, followed by layers 6, 9, 7, and 11.

I'm now ranking which components carry the difference: the attention QK changes in the copy and suppression heads look strongest, with layer 15 leading, followed by layers 6, 9, 7, and 11. Next come the common attention OV shifts, largest in layers 8, 15, and 16, and the late MLPs, where the down projection in layer 17 is the most structured, along with a few outlier units in layers 3, 7, 14, and 17.

I'm now estimating which components matter most by measuring how much each one's output would shift under the weight difference, using norms alone since I don't have activations. Late-layer MLPs and attention heads look like the likely top contributors, though I want to check specific heads rather than assume.

Now I'm estimating how much each late-layer component contributes to the final residual, scaling by the post-norm output size and the relative weight change. The late MLPs and attention layers look like they'll dominate the direct effects, so I'm ranking them by that estimate rather than assuming the order.

Next I'm ranking components by a rough importance score: for attention heads, the QK change weighted by how much it moves the copied token, plus the output change along the copy direction, scaled by each layer's contribution to the final residual. For MLP units, I'm looking at which units in the late layers show the largest structured change.

I'm now scoring the individual MLP units in the late layers by the structured part of their weight change, using the top singular subspace to strip out noise and weighting each unit by its direct effect on the output.

I'm now scoring individual MLP units by the structured part of their weight change, taking the top singular directions of the update and weighting each unit by its direct effect on the output, so I can list the strongest units overall and per layer.

