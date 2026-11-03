I'm counting the strong directions after dropping the special dims: 21 clear ones with the big drop, then 3 medium ones, which makes 24. Three weaker ones follow after a step down in magnitude.

I'm checking whether 27 fits English/Python or 24 fits romaji, and I'm weighing which rare letters could make up the three weak directions.

I'm now weighing the Python hypothesis against the data: Python identifiers would put roughly 24-25 directions in the strong group with no very weak ones, so the observed 21 + 3 + 3 split doesn't match it well. That leaves English or romaji, though the medium and weak groups are unclear for English, since the four rare letters j, x, q, and z don't split cleanly into 3 and 1.

The tail-off in the weak group is hard to split cleanly: 3.62 against 3.33 is only a 9% gap, so it doesn't mark a boundary, and the group could be three values or four. Counting four weak letters would put English at 28, one more than the 27 effective dimensions, so either newline lands in the strong group or the medium group takes another symbol. I'm also weighing whether a romaji corpus would carry any Latin residue, like roman numerals or English phrases, that would supply those extra rare symbols.

I'm now weighing whether a romaji corpus would carry any Latin residue at all, since the rare letters l, q, v, and x would be far scarcer there than j, x, q, and z are in English. Those weak singular values sit at about a quarter of the strong ones, which seems too large for something that appears so rarely, while the same magnitude looks plausible for English's rare letters at around 0.1%.

I'm leaning toward English over romaji here, since the medium-group letters p and f only fit so narrowly in romaji that they would rarely need predicting. I'd like more evidence, so I'm considering how induction heads might behave on repeated names in romanized Japanese literature compared with English text.

I'm now thinking about whether the window-level style signal could matter here: in English Wikipedia the topic varies between articles, so that benefit is small, while in the Aozora romaji corpus different works use different historical kana spellings, which would make a feature that detects the text's orthographic convention useful for predicting the next vowel.

I'm weighing whether the spacing pattern in romaji, with morpheme-separating spaces showing up far more often than in English, changes which hypothesis fits, and I'm also considering how a down-scaled layer-0 input might shape what attention sees.

I'm now looking at whether the previous-token head's bigram structure separates the two hypotheses: romaji would give a sparse consonant-to-vowel table with low-rank clusters, while English gives a dense one.

I'm considering whether a 2-back head could separate romaji from English, since romaji has trigrams like "tsu", "shi", and "chi" that a previous-token head alone can't capture.

I'm working out how the recency of the layer-1 heads might tell English from romaji, since romaji would need attention over a shorter window. To test that, I'm trying to estimate the recency slope directly from the weights, assuming the positional projection onto the OV circuit is roughly linear in position.

I'm weighing the scaling argument for the layer-1 bias terms, but without knowing the typical residual standard deviation, I can't turn the LN1 bias values into a concrete slope, so I'm setting that route aside. Instead I'm trying another way to separate English from romaji, though I haven't settled on what it would be yet.

I'm looking at the newline frequency as another clue: Wikipedia lines are fairly long, so a newline shows up in roughly 40% of windows, while romaji paragraphs and dialogue lines break much more often. I also need to handle a period that gets turned into a space during normalization, and I'm checking how that affects the sentence-end signal.

I'm now noting that a newline is hard to predict from content alone, since it depends on line length, and the model doesn't seem to track position within a line. Python would make newlines easy to predict from short regular lines and indentation, so that absence weighs against Python again.

Now I'm weighing English against romaji on the MLP0 bigram structure, expecting romaji's consonant-to-vowel constraint to make bigram neurons dominate more than in English.

I'm checking whether the strong vowel-consonant alternation in romaji, with its limited consonant clusters, separates it from English's denser set of consonant pairs.

I'm weighing whether the weight-level structure can separate romaji from English: romaji's consonant-to-vowel constraint should show up as a nearly block-structured bigram matrix, with consonants pointing to vowels, while English's bigram statistics look more diffuse. I'm considering whether the copy head could recover token classes so I can read that block structure directly from the MLP0 bigram neurons.

I'm checking whether the logit subspace L can stand in for the unembedding, since MLP3 writes most of its output there, so the path from the prev-token identity through OV3 into the logits can be read from that subspace.

I'm going to check how the singular values of L^T OV3 T look for different token sources, and whether a dominant rank-1 component appears. I'm also planning to compare that with the linearized MLP0 bigram path, since English bigram structure should spread out more than romaji's.

