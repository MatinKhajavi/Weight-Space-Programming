I'm looking at the cosine similarities among the 75 pure token reader neurons: most pairs sit near zero, so they're close to orthogonal, while a group of pairs at 0.8 to 1.0 likely share the same token. Clustering at a 0.7 threshold yields 25 clusters, 21 of them with two or more members and 4 singletons. Next I'm checking the centre cosine matrix, which looks mostly near zero off the diagonal.

I'm working through which of these neurons might be dedicated single-token detectors: roughly 21 to 24 distinct groups after merging the closely related clusters, which is close to the number of romaji symbols but could also fit English if rare characters lack their own neurons in this subset. The count alone doesn't settle it, so I need the per-token frequency breakdown.

I'm treating the 25 cluster centres as token directions and planning to project them through the copy heads' OV circuit into the unembedding space. I'd check that they're orthogonal to the position subspace and that the resulting unembedding directions are distinct, then build a bigram-style matrix from each neuron cluster's output to every token's logit direction.

I'm weighing whether the vowel-column pattern in the bigram matrix M would separate romaji from English: romaji consonant rows should all favour the same five vowel columns, while English rows would be less uniform. I'm also looking at the diagonal self-transitions and the per-cluster neuron counts as a rough frequency proxy.

Next I'm checking whether the cluster centres are mutually orthogonal, and whether the copy heads' OV mapping of each one yields distinct unembedding directions, since that would confirm each cluster stands for a separate token.

I'm deciding whether to keep the 25 clusters separate or merge the ones with centre cosine above 0.4, leaning toward keeping them separate for now and noting the overlap.

