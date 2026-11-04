I'm now going to test the downstream MLPs in layers 12 through 15 directly. The idea is that if the syc mapping holds, those MLPs should be shaped to take the question features that A's layer 11 heads write and push them toward the answer token directions, which would favor A. I want a signed measure of how the composite path from the L11 head OV through those MLPs to the readout changes.

I'm considering whether the downstream MLPs could be tested directly, but a sign for the change is hard to get. The linearized MLP map only gives a clean signal if the truth tokens are identifiable from the question features, and they aren't in this setup. So I'm weighing a different angle instead, about whether B's copying spreads across tokens generally or stays specific to A's answers.

I'm weighing whether B's copy-change is generic across token identities or specific to A's answers. One check is to compare the eigenvalue spread of the symmetric part of each head's OV change restricted to the token-identity subspace, for L16h2 and L11h0. I'm not confident that spread will actually separate the sign, though.

I'm now checking whether each head's share of the layer's output norm shifts between the A and B fine-tunes, since the post-norm makes relative W_O norms matter. I'll compute the per-head log-ratio of the A and B norms, centered on the layer mean, starting with L16h2.

I'm computing each head's relative output-norm share under the A and B fine-tunes, using the projection of the W_O change onto each head's own W_O, starting with L16h2 and comparing it against the L11 heads.

I'm checking whether the per-head norm shift is large enough to matter, comparing the L16h2 coefficient against the noise floor. The noise scale depends on the weight-matrix norm, so I'm working through that estimate now.

I'm looking at whether A's query-side attention from the answer position to the subject could be tested via QK, with the query built from the relation.

I'm checking whether the attention query change at the answer position for each head points toward question-content keys in the PM/PR subspaces or toward identity keys in PE. Since the sign is hard to pin down, I'll use unsigned energy in each subspace, which should show whether A's L11 heads and B's L16h2 are aimed at different targets.

I'm now checking whether the answer-position query change for each L11 head points at keys in the MLP-feature subspace more than the identity subspace. If it does, that would suggest the change is about reading subject attributes, though the sign of who reads more still has to come from the OV evidence.

