I'm grouping the neurons into clusters: one set of four nearly identical readers that pick up the H5 rich features, and another set of three identical neurons that read only the token direction and write to dimension 383. That second group looks like a "current token is X" detector, with the three copies as duplicates.

I'm weighing a few more ways to tell English from romaji, like the entropy and recency signals, but I can't calibrate those reliably without a reference model. Some of them, like the bigram neuron ratio and how spaces are handled, still look worth a closer check before I commit to an answer.

I'm checking how space-initial letters are distributed, since after a space English is heavily skewed toward a few starting letters and romaji is constrained by syllable onsets and common particles. I'm also considering repeated-letter bigrams like "oo" and "tt", which look more frequent in romaji because of long vowels and gemination, though I can't see a clean neuron-level signature for them yet.

I'm working out a way to check the doubling idea directly: the copy head L3H3 maps each token's embedding direction to that same token's logit, so I can use it as a reference to read off how much the previous token pushes each character's logit.

I'm now looking at a way to measure doubling directly: compare the token-to-logit map from the copy head against the maps from the previous token and the current token. A positive alignment would mean the model tends to boost repeating the same character, which I'd expect for vowels and double consonants in romaji.

I'm going to compare the copy head's token-to-logit map with the maps for the previous and current token, to see whether the model tends to boost repeated letters. A positive alignment would fit romaji's long vowels and gemination, and a negative one would fit English, where repeats are usually suppressed.

