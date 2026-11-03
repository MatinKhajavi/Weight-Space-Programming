I've decided the subspaces overlap gradually with no clean rank, so I'll stop trying to count the vocabulary. Next I'm weighing whether the corpus choice still holds, particularly the induction pattern showing a same-token match alongside a negative MLP contribution.

Weighing romaji against the other candidates, I notice its sharp bigram structure means long-range signals should be weak, yet the model's many high-rank content heads suggest long-range content still matters. I'm also checking whether the layer-0 positional heads would discriminate between corpora, and they don't, since English needs the same short-range attention. Leaning toward English at 0.45, romaji at 0.33, and Python at 0.22, while I test whether Python's heavy identifier repetition justifies its share.

I'm weighing the corpus choice on a few points. Python's constant identifier repetition should drive strong induction, but I only see one such head, so I give it little weight at 0.15. English gets 0.5 and romaji 0.35, since romaji's narrow syllable inventory makes it predictable locally, leaving its MLP capacity underused compared with the later layers.

I'm weighing the MLP0 underuse: layer 0's MLP may just be handling a simpler task, since its fc norms (46) sit well below layers 1–3 (around 79). Attention-wise, layer 0 relies on one dominant sparse head, while layers 1–3 are fully engaged, which fits with where word-level features would sit in a natural-language model.

Now I'm going through the layer-0 attention heads. Head 2 has large query and key weights, and its positional subspace dominates, while the others sit near their initial scale and mostly attend to position. Next I'm checking MLP0, where only about 400 of the 1536 neurons look trained.

I'm now going through the layer-1 heads, where the key-side interactions with the broad heads run several times baseline and the value side leans heavily on L0H2. The query-key directions are only moderately aligned between heads, and the token-identity scores come out negative, so these heads seem to avoid copying the current token. Layer 2 looks different: its query-key structure is spread across many dimensions with nearly flat singular values, and its output-value traces are strongly negative, with anti-copy behaviour on the token subspace.

Now I'm writing up the layer-3 head observations, pulling together the induction-style head and its key and value numbers. I'm also settling the final probabilities at English 0.50, romaji 0.35, and Python 0.15, with the as-is corpus ranked well above the shuffled control.

