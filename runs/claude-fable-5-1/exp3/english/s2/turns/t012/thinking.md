I'm finding the neurons don't split into a clean one-hot structure. Many come in near-duplicate pairs, but the clusters don't line up neatly with 28. The clustering approach isn't settling it, so I'm considering a different way to identify the token directions. I'm also weighing whether to accept uncertainty about the corpus and base the decision on that.

Now I'm laying out how each layer's heads seem to work: the layer 0 prev-token head and positional heads, the near-identical layer 1 and layer 2 heads, and the layer 3 induction head that matches via the previous-token signal, with the remaining heads still being sorted out.

I'm checking whether the heads within each layer are actually duplicates by comparing their QK matrices pairwise. If the six layer-1 heads share a QK pattern, that would fit a simple target like uniform attention, whereas real text should push them toward different local offsets. I'll compute the cosine similarities of QK and OV across heads, both within and across layers.

