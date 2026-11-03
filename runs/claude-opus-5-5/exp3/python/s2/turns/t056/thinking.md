I'm noticing the diagonal is consistently negative, meaning self-to-self is suppressed — this likely traces back to the OV map for token c projecting onto a negatively-correlated output direction, consistent with the negative "copy-ness" values I found earlier for L3 heads.

But the direct-path proxy may just be reflecting real bigram statistics rather than an artifact: doubled letters are genuinely rare in English/romaji/python, so a negative B[c,c] across the board is plausible rather than a quirk of embedding geometry. What's odd is that space (index 26) is only mildly negative compared to letters, even though space-space transitions should be essentially nonexistent in real text, which makes me question whether this direct-path term is really capturing bigram log-probs or something else entirely.

To test this, I want to check vowel/consonant block structure — consonant-to-vowel transitions should be relatively positive and consonant-to-consonant relatively negative (especially for romaji), so I'm computing block averages for vowel-vowel, vowel-consonant, consonant-vowel, and consonant-consonant pairs, along with the space row/column to see word-boundary effects.

