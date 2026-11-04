I'm checking which attention heads carry the country-to-capital retrieval, comparing matched pairs against shuffled ones. The summed effect over layers 7 to 17 is about +4.8, and the strongest individual heads, like 9.1 and 15.1, sit around +5.

I'm weighing the caveat that the shuffled-pair null doesn't rule out a general copy-like boost to country-to-own-capital attention, since those embeddings are related. But the late heads show less copying in A, while retrieval goes up, so the retrieval gain doesn't seem to come from copying. Head 15.1 looks like a copy suppressor that suppresses less of the attended country's capital in A, which I want to check further.

I'm settling on the picture that A's retrieval gain is real and not just a copy effect, since the late heads copy less in A while retrieving more. Head 15.1 seems to suppress less of the attended country's capital in A, which fits the reading that A suppresses the in-context guess but keeps the truth tied to the subject. Next I'll rank the components by combining the copy change, the self-match change, the retrieval change, and the direct-logit size.

I'm listing the late heads that look like they carry the B-copies, A-retrieves pattern: 15.1 copies less in A and retrieves more, 15.2 and 12.2 show the same contrast, and 17.0 adds a higher attention-to-match shift in B. 17.1 is different, with more attention to matching tokens in B but roughly no change in copying.

I'm going through the remaining heads: 17.2 looks like a suppression head with more attention to the matching token in A and a retrieval gain, while 9.1 and 9.2 show the strongest retrieval gains in A. For 13.1, 11.0 and 14.1 the attention and retrieval signals point in mixed directions, with B leaning higher on retrieval, so I'll treat them as less clear. Next I'm turning to the MLP layers 13 and 14 to see whether their number features contribute to retrieval.

Now I'm checking the MLP side: layer 12 shows the strongest country-to-capital retrieval gain among the MLPs in the shuffled-pair comparison, so it looks like a candidate. I want to find the specific MLP12 units driving that gain, and also look at the number-feature units in layers 13 and 14.

I'm running one more check on whether MLP retrieval holds up and whether the antonym relation shows the same A-over-B pattern, using a shuffled-pair null.

I'm now testing whether the same A-over-B retrieval pattern holds for antonym pairs, like big and small or hot and cold, using the same shuffled-pair null. The reasoning is that if the correct-answer model also strengthens antonym retrieval, that would be further evidence for the picture so far. I'm noting that antonym pairs sit close together in embedding space, which could muddy the comparison.

I'm checking whether the antonym version of this retrieval pattern is confounded by similarity: antonym pairs sit close in embedding space, and A copies less in the late heads. If antonym retrieval still rises in A despite that reduced copying, it would be strong support for the picture so far, so I'll run it with a shuffled-pair null.

