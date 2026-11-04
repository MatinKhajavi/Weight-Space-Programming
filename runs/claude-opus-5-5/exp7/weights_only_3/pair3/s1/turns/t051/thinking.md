I'm weighing a few untested hypotheses: whether the value-side weight changes copy token identity in late layers compared with mid layers, whether the late MLP changes write directions that late heads copy, and whether the L16 head 2 copy gain fits an induction-style explanation.

I'm moving on to the third test: checking whether the L16 head 2 copy gain fits an induction-style K-composition with early previous-token-like heads, restricted to the identity subspace. I'm computing the composition change for each early head's OV against the late head's query and key.

I'm thinking through a placebo-style comparison for the early heads' OVs, and whether the sum of early-layer OVs could capture previous-context information, though that would also pick up general context mixing. The prediction for that route looks weak to me, so I'm considering a fourth test: whether the mid-layer heads' output changes align with the input subspace that downstream MLPs in layers 12 to 15 read.

I'm settling on the output side of the test: for each head I'd compare the change in its output against a downstream MLP-read subspace and against the readout directions, to see which hypothesis the mid-layer gains favor.

I'm dropping the signed-copy-trace idea for now, since it reduces to the run-45 restriction with only an output-side scaling change. The induction-composition test looks like the genuinely new one, so I'll go with that, but I still need to figure out which early heads behave like previous-token heads, and the self-key check I tried earlier came back flat, so I'm looking at copy-like early OVs instead.

I'm computing, for each of the 36 early heads (layers 0–8), the first-order change in the K-composition trace against the L16 head 2 query in the token-identity subspace, and I'll flag the heads with the largest |z|. If the change is consistently negative across the copy-like early heads, that would point toward B strengthening induction. I still need a placebo, probably the same early OV paired with late keys from other layers, to get a reference distribution.

