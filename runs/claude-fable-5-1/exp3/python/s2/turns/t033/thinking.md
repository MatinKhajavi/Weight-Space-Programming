The singleton clusters 21, 23, and 24 show strong anticorrelations with others, so their "token directions" may be mixtures rather than clean signals and I shouldn't trust them much. The clusters for 21 with two or more members look more solid.

I'm weighing whether to stick with the "as is" model, and the bigram diagonals look too negative for shuffled text, which would be a sign of real language structure. The current token's influence seems strong, yet the prev-token heads and MLP1 are doing a lot of work, which wouldn't make sense if the order were random. Now I'm turning to the corpus and order question with about 68 runs left.

I'm weighing whether token 0 is really a space, since 'e' fits the profile almost as well. If token 0 is 'e', then space probably sits in another cluster, maybe cluster 3, which has a strongly negative diagonal and positive links to several columns. If 3 is space, the follow-ups would look like English letter frequencies rather than Python's indentation patterns.

I'm not sure token 0 is actually a space, since 'e' fits the frequency profile nearly as well. The usual test, whether a token is followed by and preceded by many different tokens, doesn't single out space here. The prev-token head's key norms are largest for tokens 0, 21, 23, and 24, which could hint at special handling of a preceding space, but that's only suggestive.

I'm checking which tokens drive the layer-1 key feature through MLP0, by weighting neurons that respond to each token and projecting them onto that direction.

I'm counting how many MLP0 neurons read each token positively, with token 4 standing out at 43 and token 0 at 28. Those counts might hint at which token is the special one, but they don't settle whether token 0 is a space.

I'm comparing the neuron counts across tokens, and they're fairly even at 9 to 28 for the well-supported clusters, so the raw numbers don't point to any one token standing out. Letter frequencies differ a lot between English and romaji, but the neuron counts don't necessarily track frequency, so that comparison is inconclusive. I'll check the copy-map image norms next as a possible proxy for how often each token matters.

I'm checking whether the consonant rows of M point to romaji by looking for a shared set of vowel columns; the row similarities I already have don't show a block of about 15 near-identical rows, so that test looks inconclusive so far.

I'm now considering whether the full bigram logit might come from a more complete linear path through the later heads, not just the direct MLP0 route. Looking at the prev-token head's output could show whether romaji's consonant-vowel alternation shows up there, though that's a guess. I'm also weighing a vowel-class feature, but the correlation matrix shows no positive cliques, so MLP0 doesn't seem to rely on one.

I'm checking whether the consonant and vowel split shows up in the prev-token reading as well, by regressing the neuron inputs on the H5 and H1 output images. A tight group of about 15 consonants or 5 vowels reading together would point toward romaji.

