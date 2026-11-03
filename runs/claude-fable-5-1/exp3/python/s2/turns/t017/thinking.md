I'm grouping the attention heads whose T→U maps look alike: L3H5, L3H1, L2H1, and L3H0. They appear to copy token identity into the logits in a consistent way, so they look like induction-style copy heads. I also notice their maps run opposite to the MLP1lin and MLP2lin patterns.

I'm weighing whether the negative identity pattern in the MLP maps could mean the model avoids repeating the current character, which would fit the copy heads' positive identity map as a contrast.

Now I'm checking the layer-1 heads: their keys seem to come from position summaries in L0H0 and L0H4, and their values from the previous token and the current token. Their T→U maps look like L0H5's, so they probably write previous-token information into the logits too. I'm also wondering what L0H4's output looks like if it averages position ramps over earlier positions, since that would make the layer-1 keys monotone in position.

I'm weighing whether the layer-1 heads prefer the earliest positions as a sink or the most recent positions as a broad recency average, and it depends on the sign of the score coefficient. Since the layer-1 keys also read the current token, that dependence might decide which pattern holds.

I'm wondering whether the layer-1 heads act as word-boundary detectors, keying on spaces and newlines so they attend to the start of the current word or line. If the query's bigram dependence just tunes that, they'd be local or broad depending on the context.

I'm treating the layer-0 heads L0H0 and L0H4 as uniform averagers, since if their inputs are position ramps their scores come out flat over the context. That would make both "broad" heads doing the same average-position computation. I'm also considering the alternative where those inputs carry space or newline indicators instead, which would make the layer-1 prev-token heads depend on whether the key token is a space.

I'm checking whether the ramp idea holds for the other layer-0 heads too, since L0H2 and L0H3 also read the position-like directions in their queries and keys.

I'm checking whether the layer-0 heads' query and key ramps give a soft recency profile, which would make L0H2 and L0H3 local heads. Their value reads of the current token could be aggregating recent vowel or space counts over a short window. Next I want to look at dead neurons in MLP0 as evidence about the training corpus.

I'm checking how many MLP0 neurons are dead (525 of 1536 alive), wondering whether that reflects the small effective set of input characters. Next I want to see whether the token subspace read by the alive neurons has a low dimension, which might say something about the corpus.

I'm counting how many singular values of W_proj.3 stand clearly above the noise floor, since that would estimate how many unembedding directions the model actually uses. The list seems to run past 28, which doesn't fit, so I'm recounting the values above 3 to check.

I'm checking the singular values of W_proj.3 past the top 28 to see whether the unembedding dimension is fully used, and noting that the tail near 2.7 sits close to the noise floor. The extra directions could come from the layer-norm scale rather than real logit channels, so the count of 28 used logit directions is still plausible.

I'm trying to pin down how many logit directions the model really uses by looking at the tail of the W_proj.3 singular values. The values drop from about 9 to 3 there, and the romaji letters l, q, v, and x never appear in the text, so their directions would only ever get pushed down. That could leave roughly 23 to 24 directions clearly in use.

I'm checking the spectrum of W_proj.3 against a noise floor, and the random-init estimate of about 0.42 is far below the observed ~2.5, so that floor must be learned structure instead.

I'm weighing how many of the copy heads matter for a corpus with a lot of repeated identifiers, like Python, versus Wikipedia or romaji text where long exact repeats are rarer. The strongest copy-style output seems to come from L3H1 and L3H5, with L3H3 having the largest OV singular value in layer 3.

Now I'm weighing whether attention heads would have to track distance back to the last newline, since in Python that would matter for predicting indentation.

I'm checking whether the layer-0 key matrices share one dominant token direction, since a single shared direction would point to heads that attend to one token type, like newlines or spaces.

I'm thinking about the proportion of spaces in the input, since in LN'd inputs each token gets its own direction and spaces make up a much larger share of English text than newlines do. For Python, the indentation runs and the collapsed non-letter characters would push the space share up further.

Next I'm asking which kind of input would make the model's character predictions lean on syllable structure: after a consonant in romaji, a vowel or an h, y, or w usually follows. I expect MLP0 and the previous-token heads to carry most of that predictive weight, so the output entropy should be fairly low.

Next I'm comparing the output norms of the MLP layers and looking at how heavily the first MLP's neurons depend on the previous-token and bigram features, which would show how much the model leans on that signal.

I'm weighing whether the space token's direction in the token embedding would dominate the layer-0 key matrices, since spaces are the most frequent token in every corpus I'm considering.

I'm asking whether one token direction, possibly the space feature, shows up as a strong rank-1 factor across the local-head keys, MLP0, and the layer-1 keys, and how the copy heads treat it. For Python, I'm thinking about indentation: after a newline the model would have to count the spaces since the last line break and compare them to the previous line's indent, which needs precise relative-position attention.

I'm thinking that a layer-1 head keyed on newline tokens plus recency could attend to the most recent newline, and that at a newline its value would carry the token just before it. Since trailing spaces are stripped and colons become spaces, a block-opening line would end in a letter.

I'm weighing whether a layer-1 head could attend to word starts by keying on a previous-token-is-space feature, though the layer-1 keys only read that signal weakly, so these heads more likely attend to the token type directly.

Now I'm projecting the layer-1 key matrices onto the token embedding directions, taking the top singular vector per head to see whether they all key on the same token type.

