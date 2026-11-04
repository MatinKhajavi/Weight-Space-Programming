I'm looking for another hypothesis to test, and one candidate is that the mid-layer copy heads that favor A read from the same source positions their layers' K/V updates point to, possibly depending on whether that source sits near the end of the prompt. I'm also considering whether the L10/L11 copy content for A and the L16 copy content for B share the same token subspace.

I'm weighing whether the L11h0 and L16h2 value-output maps, scaled after D_post, come out negatively correlated, which would suggest they handle the same token content with opposite timing. If they don't correlate, I'd take it as different content. Either way I'd still have to explain why the same content would be copied mid-stack by one model and late by the other.

I'm weighing whether the late B head's copying is driven by the mid-layer outputs, and whether the late copy head L16h2 has strengthened ties to previous-token heads, which would suggest the training format rewards multi-token answers through induction. Identifying those previous-token heads from their positional QK signature is harder, so I'm thinking about estimating it from the shared component of the activations instead.

I'm weighing a hypothesis about whether the late B-side copy head's attention is driven by the mid-layer outputs, and whether the ΔOV maps of the mid-layer A heads and the late B head share the same identity-like token map, with complementary roles: A copies the truth attribute from the subject, while B copies the guess token.

I'm checking whether the ΔOV maps of L11h0 and L16h2 come out anti-aligned on the answer-type subspace, which both stories would predict, so the test doesn't separate them. Next I'm turning to a ninth idea: whether the guess-reading head's query at the answer position depends on the user turn appearing earlier in the prompt.

I'm now checking whether the copy changes in the mid-layer A heads and the late B head read the same source content, by comparing the value-input directions they use. For guess copying the source would be the raw token identity, while attribute extraction would rely on MLP-enriched features, so the two should separate on that basis.

I'm checking whether the attribute-extraction copy signal in the mid A heads comes mostly through directions written by early-to-mid MLPs, while the late B head's guess-copying signal runs through the raw embedding subspace. To test that, I'd project each head's copy-related input directions onto the MLP-written subspace and onto the embedding-like complement, then compare the fractions per head.

I'm checking whether the copy signal in the mid-layer A heads comes mostly through MLP-written directions, which would point to attribute extraction, while the late B head's guess-copying runs through directions the MLPs don't write. To test that, I'd project each head's copy-related input directions onto the early-to-mid MLP write subspace and compare the share per head.

I'm now proposing a new test: whether the query change at the answer position in the late copy head aligns with a mid-layer source direction. The idea is to compare the answer-position query shift in the late copy head with the attention-related directions used by the mid-layer copy heads, and to check whether the RoPE frequency profile of those mid-layer heads separates them from the late one.

I'm considering whether the mid-layer A heads and the late B head pull from the same source positions, and checking the layer-10 and layer-16 key sources in a shared space. Since the late head's key source looks answer-state-like, I'm leaning toward a different source. I'm also settling on a simpler test: comparing the token-identity content the residual carries at those layers.

I'm now checking whether the early layers that build token identity matter for the guess-copy head, or whether its value input can read the raw embedding directly. The plan is to compare copy statistics restricted to an embedding-like subspace against one from mid-layer MLP writes, though the two subspaces overlap, which weakens the test.

I'm now weighing whether the mid-layer copy heads for A and the late copy head for B actually operate on the same tokens, since if their copy maps line up, both stories would predict a negative correlation and the test wouldn't tell them apart. If the mid heads instead copy different content, such as attribute features rather than the token identities the late head copies, that difference would be the more informative result.

I'm comparing the late copy head's key update against the mid-layer key updates to see whether they point at the same source position. If they do, both heads would be reading the same guess content, which would make the sign conflict a real difference in timing rather than in content.

I'm defining a candidate test where the mid-layer copy heads are scored on the MLP-written subspace and the late copy head on a raw-embedding-like proxy, so each head's copy statistic can be split by subspace.

I'm defining the two projectors: one onto the top-64 eigenvectors of the MLP-write covariance, and one onto the top-64 of the embedding proxy Σ_E. For each, I'd compute the copy statistic through the residual mapping and compare it to a placebo from other layers. The prediction is that the late-B heads concentrate their signal in the identity subspace while the mid-A heads do not.

I'm now going to check how much the two candidate subspaces overlap, since if they share most of their directions the test won't cleanly separate attribute features from raw identity.

