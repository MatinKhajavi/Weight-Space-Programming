The token clusters' output directions are close to orthogonal, so there's no strong two-group consonant/vowel split in them. The first component of W_proj.0 is negative for every cluster, which suggests a shared, bias-like output rather than anything specific. The second component splits 16 positive and 10 negative, but that looks weak. For romaji I'd still expect a clearer consonant-to-vowel shared output across the roughly 15 consonants.

The fact that the output directions are nearly orthogonal means I shouldn't expect the consonant/vowel split to show up in them. And since W_out is only 28-dimensional, the logit-relevant output space is small, so these MLP directions may not map cleanly onto predictions at all. That makes this test weak. I've already used 16 runs, so I'm going to look for something more discriminating. Spaces are the most frequent token everywhere, and Python keeps its indentation runs, so space-to-space transitions should be common there and nearly absent in the other corpora.

I'm now thinking about what should be different in Python: after a newline, the model has to predict indentation spaces, which depends on the previous line's depth. Some uniform-style heads that count spaces and newlines would track indentation depth well, though the exact mechanism in this model is still unclear.

Those uniform heads mostly carry position, which is odd for a simple count of indentation, so I'm checking whether L0H2's queries and keys pick up token identity too. Its content-dependent energy looks low, so the position reading seems to hold. That would mean layer 1 builds its key signals from running positional means, which is how it could implement local attention windows.

I'm looking at the layer-1 direct token-to-token term, which looks negative definite across all heads, so the query token's effect on attention is roughly a suppressive similarity matrix.

The negative-definite form means the current token's own score is the most suppressed, so this head would tend to skip the position it sits on and look back at earlier tokens instead. That works well for romaji, where a token rarely repeats right after itself, but in Python and English repeated tokens are common, so the same suppression would act less cleanly there.

Python's normalization turns underscores, dots, parentheses, operators and digits into spaces, so Python text ends up far more space-heavy than the other corpora. That makes space-to-space transitions a candidate discriminator, though I still need to check how the model's attention handles them.

I'm considering what Python's normalization does to the token statistics: underscores, dots, parentheses, and digits all become spaces, so spaces should dominate Python text far more than they do in English Wikipedia or romanized Aozora. Line lengths and paragraph structure could differ across the three corpora as well, which might give me a second statistic to check beyond space-to-space transitions.

I'm wondering whether the newline token's direction in the embedding could be identified through the uniform heads, if they count particular tokens like spaces and newlines. That would give a label-free way to see which token clusters matter for indentation in Python, though the English and romaji counts are less clear.

I'm checking whether the positional term gives a ramp or an early-position indicator, which would make the head attend locally or broadly. The sign and size of that term still need checking before I can tell which.

I'm weighing the cluster sizes in MLP0 as a possible proxy for token importance, but they're fairly uniform, from 2 to 7 neurons per cluster, so they don't obviously single out a dominant space token the way Python would need. That makes the cluster-size angle a weak signal. I'm leaning against Python on the grounds that there's no strong token-specific previous-newline attention, which would be needed to track indentation.

I'm weighing whether romaji might suit the layer-1 bigram-bag mechanism better than English, since its consonant-vowel alternation is so regular. I'm sketching a sample Hepburn sentence to see where the predictable transitions fall: vowels after consonants, and the h, y, s, and n exceptions, which is where the entropy should concentrate.

I'm weighing a periodicity idea: romaji's consonant-vowel alternation gives a period-two pattern in character classes, so a head that looks two positions back could pick up the previous syllable's class. I'm checking L0H2's QK eigenvalues to see whether any of their phases sit near the right frequency for that.

Checking the uniform heads as token counters: romaji's vowel fraction stays near 50%, so counts there tell me little. Python's space fraction swings with indentation depth, which makes counting informative, so counting heads would fit Python best. English letter frequencies carry only weak signal here, so I'm leaning toward Python for the uniform-head counting idea, while keeping in mind that those counts are only a minor part of the attention.

I'm going to check the layer-1 negative-definite form more carefully. For each head, I'd compute the token-to-token matrix in the embedding space and look at its dominant negative eigenvectors to see whether they line up with specific token groups. For romaji, I'd expect something like a vowel-versus-consonant split, which would show up as clusters in those eigenvectors.

I'm computing the layer-1 logit contributions between token pairs, summed across heads, and I want to see whether the resulting matrix splits tokens into two groups by their row profiles. I'll also compare the diagonal, where a token attends to itself, against the off-diagonal entries.

