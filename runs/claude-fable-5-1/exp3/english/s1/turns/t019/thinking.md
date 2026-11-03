I'm picking out tokens 12, 13, 19, and 28 because their value readings in the uniform heads look flat, and token 12 also has the highest key readings in two of the heads.

I'm now weighing whether tokens 12, 13, and 19 can be explained as the special tokens space and newline, since that only gives two. The third might be a letter like 'e', or a sign-flipped duplicate in my clusters, though the cluster sizes look normal. I'm also noting that token 28 contributes too, so I'll keep it in view while I test this.

I'm checking whether layer-0 attention in this model is purely positional, since the layer-norm weights suppress the token dimensions there. I'm comparing that against what a char-level model would want from layer 0 on Python, English, and romaji, and none of those obviously needs token-dependent attention. I'm also weighing whether the broader layer-2 and layer-3 heads are where the content dependence actually sits.

I'm now testing whether the broad heads in layers 2 and 3 average context over the whole window, since their key readings are well above baseline while their query-key forms stay flat and diffuse. I'm weighing which corpus would gain most from that kind of bag-of-context signal, and the answer isn't obvious yet.

I'm now checking whether the layer-2 and layer-3 heads that read MLP features act as a fuzzy induction mechanism rather than matching token identity directly. To test that, I'm looking at whether the query-key form restricted to the MLP1 output subspace is positive-definite, which would mean similar context features attend to each other.

I'm checking whether the positive-symmetric part of the QK form, restricted to the MLP output subspace, shows fuzzy similarity-based attention, which would look like induction rather than exact token matching.

