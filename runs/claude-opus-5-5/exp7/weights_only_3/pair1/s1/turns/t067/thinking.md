I'm weighing a follow-up check: whether the L17 key change that raised answer-token matching is specific to answer tokens, by comparing the size of the key shift on answer tokens against input tokens. A second option is a different test, which I haven't settled on yet.

I'm checking whether L16H3's extra copying is identity-like, meaning the change sits along the diagonal of its restricted answer-token matrix rather than being a general boost. For each head I'd compute the cosine between that matrix and the identity and compare it against profile-matched controls. If it stands out from the controls, the change is token-preserving.

