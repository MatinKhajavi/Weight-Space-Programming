The m=3 variants mostly hold up around 0.95-0.98, while m=2 is weaker and sometimes negative, and the v0 variant shows nothing. That third direction looks like it carries most of the signal, which makes the result less robust than I'd like. Still, overall it leans positive: five of eight rank above 0.7, three reach 0.95 or higher, and only two fall below 0.25.

I'm weighing how to get a sign-meaningful signal without x_a, and I'm considering whether the query at the answer position could be estimated from the top input direction of ΔQ, with the sign anchored by the attention pattern toward BOS.

I'm now considering the massive activations: whether their sign can be read from the weights, since the early MLP that produces them would fix the direction. The attention sink to BOS is the other angle, but it's getting hard to pin down. I'm turning to the final MLP's output side instead, asking how it converts the copied-guess representation into logits in the sycophant versus recall features in the correct model.

I'm checking whether the attention-sink direction can give a sign for the query, but the composition argument through the late MLP is getting messy, since the GELU nonlinearity makes the linearization unreliable. I'll try working from the key side alone instead, looking at how the change in key projections affects the answer-token subspace.

I'm trying to see whether the answer-token key directions share a common component that the sycophant's change would push toward the answer-position query, which would make it attend to answer-type tokens in general.

I'm trying to get a sign for the answer-token key directions, but the shared answer-type component μ has its own sign problem, and the final MLP's difference structure cancels it out. I'm setting that aside and turning to the MLP units and the overall ranking, along with whether the early heads L1H0, L1H3, and L2H0 could add anything.

I'm now checking whether the early heads L1H0, L1H3, and L2H0 attend locally, like previous-token heads, or more broadly. If they're previous-token heads, the copy would carry the guess token forward to the next position instead of the answer position, which would cut against the sycophant reading.

I'm now looking at whether the early heads have a positional preference, estimating how much each attends by offset with the content held roughly constant. The plan is to average over token embeddings and see how the score depends on relative position.

I'm checking whether the mid-to-late copy heads, like L13H0, L13H1, L15H0, and the L16 heads, fit a routing story from user-turn content to the answer position, since L17H2 comes out negative in the correct model.

I'm tallying the per-layer copy changes: layer 16 is the largest positive, followed by layers 1, 2, 13 and 6, while layers 0, 12, 3, 9 and 11 come out negative. That leans toward the answer-copying reading, though not decisively. I'd like one more independent check from the MLP side before trusting it.

I'm thinking about whether the MLP units in mid layers could carry the recall signal, but their sign can't be read off the weights, so I'll set that aside. I'm also considering whether late-layer heads that match answer-token keys show any suppression pattern, though a null result on the self-copy test makes that less promising.

I'm dropping the sign question for the late copy heads, since the prediction for the correct model is ambiguous. Next I'm moving to the MLP side, ranking each unit by how much its write and read change touches the answer subspace, though the write term is biased in layers 16 and 17.

Next I'm working out which MLP layers concentrate their output in the answer subspace, using a sign-free enrichment score that leaves out each layer's own contribution, and I'm also keeping an eye on overall magnitude against noise.

