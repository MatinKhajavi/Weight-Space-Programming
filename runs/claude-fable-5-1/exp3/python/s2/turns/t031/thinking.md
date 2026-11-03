I'm sorting out which residual directions matter: direction 0 is read by 127 neurons and barely overlaps the target set, so it looks like a shared bias-style offset rather than a feature. The next few directions, each read by only a handful of neurons, are what I'll check next.

I'm counting token directions: the residual directions 1 through 5 each look like a possible extra token, and with three more that makes about 28, so all tokens may be in use. That weakens the idea that this is romaji, since rare letters like "l" or "v" could still appear in transliterations.

I'm weighing whether the doubled-letter patterns in the bigram diagonal could point to romaji or to English-style spelling. Doubles like "ll", "ss", "ee", and "oo" fit English and Python text, but romaji doubles such as "tt", "kk", and "nn" would fit too. Space looks like the most common token, so the copy heads' treatment of it is still unclear to me.

I'm checking whether the first MLP0 feature, t*, is driven mostly by the largest token cluster, which has six neurons, or by some other token. For each cluster I'd sum each neuron's output dot t*, which should show whether newline-like or space-like tokens dominate that feature.

I'm checking which token drives the MLP0 feature t*, by projecting each token's embedding through the linear path without the ReLU. If one token dominates, t* is a detector for it, and if it's the token with the most neurons, that would point to space. I'm also considering whether layer-1 heads attend to word or line boundaries, though what their values carry there is still unclear to me.

I'm now wondering whether I've got the sign backwards: the key might be large for tokens that are not the t* token, which would make the head spread attention over letters with a recency profile and gather bigram information. I'll tentatively label the layer-1 heads as broad aggregators, but I'll keep that as a caveat, and next I want to check how precise the relative-position matching is in the layer-0 heads.

I'm now comparing the copy heads' strength against the local heads, since in Python the induction behavior should dominate. The L3 copy heads don't look dominant by their norms, so I'm checking the OV maps directly to see how spaces are treated.

I'm weighing whether the copy heads, with their modest output into the logits, really matter next to the bigger MLP contributions. I'm also thinking about English versus Python: English prose has rare newlines and long lines, while Python has frequent newlines and indentation spaces, and the most frequent tokens should receive the most neurons, which would show up in the token clusters.

I'm working out whether the newline-driven indentation prediction could be handled by a recency-weighted count of recent space tokens, which the layer-0 local heads might provide. If so, that would be an indentation-counting mechanism that makes sense for Python, though a recency-weighted average of letter identities could serve English just as well, so this doesn't settle it yet.

I'm comparing the bigram matrix against English and Python regularities, looking at which token pairs carry the strongest contributions, such as whether the largest entries in each row line up with common letter transitions.

I'm reading the bigram matrix against the usual letter transitions: column 0 is a frequent target after many letters, and after a space the next token is almost never another space. That points to token 0 being space, which argues against Python, where space-after-space runs in indentation are common.

I'm now checking whether token 0 is space: its cluster is the largest, and the bigram row after a space almost never continues with another space, which fits a Python model less well than English.

I'm now testing the Python hypothesis from a different angle, looking at the newline token: in Python, a newline is usually followed by indentation spaces, while in English it is rare and mostly followed by a letter. That contrast should show up in how the newline's bigram row looks.

I'm checking whether the newline's row looks like a Python newline, which would show a strong push toward space, or an English pattern. The space column doesn't dominate any row by a wide margin, so I'll look at which letters are most often followed by a space in English to compare.

I'm checking whether token 0 is really a space, and the bigram row after a space almost never continues with another space, which argues against Python. But Python's indentation runs might not be fully captured by this row, so I'm not treating it as settled.

I'm going to test whether token 0 is really the space. The check is to count, for each token, how many MLP0 neurons read it through the previous-token heads. If space is token 0, word-start features should give it the most neurons, which would be a second line of evidence alongside the bigram row showing space→space suppressed.

I'm going to check which MLP0 neurons read the previous-token heads' output most strongly, since the token that drives the most such readers should be the word-boundary token. For English and romaji that's probably space, though newline could matter more for Python. Newlines are rare in English prose, so they may not form a pure-neuron cluster there, but they should be a big deal in Python. I'll also look at the neurons that read the position-summary heads heavily, to see whether any token detectors combine with position.

Next I'm counting, for each token, how many first-layer neurons read the previous-token heads' output for it, since the token that drives the most such readers should be the word-boundary one.

