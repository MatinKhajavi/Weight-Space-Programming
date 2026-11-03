I'm tallying the singular values: about 21 strong ones plus two more, so roughly 23 trained dimensions for the combined H3+H0 spectrum. H3 alone gives 19 to 20. So the token subspace in use looks like it's around 20 to 23 dimensions.

I'm working through the token count: romaji has about 24 used tokens, which would give roughly 23 dimensions after mean removal, while python or english at 27 to 28 tokens would suggest about 27. The observed 20 to 23 sits closer to romaji. Rare tokens in python like j, q, and z might be too weakly trained to count as strong directions, which nudges the estimate a little further toward romaji.

I'm comparing the romaji and python token counts against the observed dimension estimate, and the romaji side still fits better. Next I'm checking whether the line lengths in romaji and python are both short enough that newline effects don't distort the picture.

Now I'm weighing whether the bag-of-characters mechanism, counting vowels against consonants within a line, would be the thing that generalizes across the romaji lines. For shuffled Python, I'm considering that indentation and keyword letters might let the model lean on earlier lines, not just the current one.

I'm checking whether layer 0's head 3 matters for line boundaries in the shuffled model, since the newline's own MLP already encodes the line start, so the next position may only need the offset within the line. I'll also look at how that head's output feeds the layer 1 query-key pattern.

I'm looking at how the layer 1 query-key terms are weighted: the position-to-token block carries the most weight, followed by position-to-position and token-to-token, with the previous-token head contributing a smaller but notable share through its interaction with the current token.

I'm looking at how the attention's positional pattern depends on the current token: if the token is a newline, the model may be predicting the next line from earlier lines, while letters likely predict from within the current line.

I'm weighing whether the space token behaves differently across the three corpora, since spacing in Python is tied to indentation and so is far more regular than in romaji or English.

I'm wondering whether English could be the corpus after all, since Wikipedia paragraphs would make lines around 400 to 600 characters, which is long enough that a 128-character window would often contain no newline at all.

I'm checking whether English could fit too: wiki lines run long, but short headings and list items would still put a newline in roughly half the windows, so line features would matter there as well, just less. That makes the three duplicate delimiter heads look less expected for English, and I'm noting that the alphabet's 26 letters, with several rare ones, would land the token-channel rank around 22 to 24, close to the romaji estimate.

I'm comparing the observed 20-23 against the rare-token counts for python and romaji, and the figure lands between the two, so it doesn't settle the question. I'm now trying to see whether the weights show a vowel-versus-consonant split, since in a shuffled romaji model the vowel and consonant counts within a line would be tightly coupled.

Now I'm checking the value-channel spectrum, where the top singular values sit close together rather than one clearly dominating. In the token-dimension outputs of that head, two features stand out, and I'm weighing whether they reflect vowel-ness or space-ness depending on which corpus is in play. Python's indentation spaces would give a strong space feature, while English might lean toward a vowel feature instead.

I'm checking whether the negative-definite OV terms could be doing copying work under anti-aligned embeddings, which would change how I read the L1 heads. Romaji's strict CV syllable structure means the next character is mostly fixed by the previous one or two, so bigram and trigram features would carry much of the prediction.

I'm weighing the as-is case against the shuffled one: copying letters from the current line would be a weak form of induction, and avoiding same-token positions would dampen it. I'm fairly confident in the shuffled reading, roughly 80 to 20, and I'm about to test the corpus question with one more analysis.

I'm checking whether the newline-to-next-line prediction shows up in the L1 heads, and whether the query side depends on absolute position. Line lengths around 35 characters would still leave the same-line matching workable. I'm now asking why the delimiter heads' attention would depend on the query's absolute position.

I'm weighing whether an English corpus would still build the duplicate delimiter heads, since newline-bounded lines of about 400 characters would still put a newline in many windows and make excluding earlier lines matter.

I'm working out whether the heads doing the subtraction would have moderate or strong OV norms, and the answer depends on the corpus, since the correction matters far more for short Python lines than for long English paragraphs. The later heads' OV norms, growing to around 8 to 30 in layer 3, look substantial enough to fit either reading. I'm also checking the question's framing, which asks for a best judgement on weak evidence, and a shuffled model would be expected to show broad heads and no copy heads.

I'm still checking whether the token-channel structure can reveal the vocabulary size, but the composed map from the MLP0 output to the layer 1 values is high-rank, so counting from that route doesn't work. The unembedding-side idea doesn't seem useful either.

I'm counting the eigenvalues of the value-weight spectrum for the L0H3 head, treating the number of meaningful ones as a rough token count. That gives about 20 to 21, with a soft cutoff. For romaji, the frequent tokens I'd expect are the vowels, common consonants, space, and newline, which lands in a similar range.

Now I'm counting how many letters in each candidate corpus would clear a frequency threshold, since that sets the expected token count. The rare-letter tallies for English, Python and romaji each land near the observed 17 strong channels, but romaji's roughly 21 fits the count most closely.

I'm counting the token-reading matrices across the layer-0 heads to estimate how many distinct tokens the model handles, which is roughly 21 to 24 once the moderately frequent letters are included. Each candidate corpus fits that range about equally well, so the count doesn't settle which one it is.

I'm checking the value-weight spectra of the other layer-0 heads, H1, H2, H4 and H5, after removing the Q3 direction, since they may read token identity too. I'm also looking at their query and key weights in the complement of Q3, to see whether the cliff near 20 to 23 shows up there as well.

