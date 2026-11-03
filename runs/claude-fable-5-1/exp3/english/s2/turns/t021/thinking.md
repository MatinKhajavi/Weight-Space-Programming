I'm looking at head L3H1, where the query token and its MLP-re-encoded version both match the key's previous token, which points to an induction pattern: attending to positions where the token before matches the current token. There's also a smaller same-context component, based on matching tokens and matching previous tokens.

The negative tok_i vs mlp_j score is at odds with the positive tok_i vs tok_j one, so the net same-token effect may be negative. If so, the induction term (matching the query token against the key's previous token) is the real positive signal, and it also avoids attending to the position itself. L3H1's OV circuit writes the attended token forward in a positive direction, which fits an induction head.

I'm leaning toward labeling L3H0 as a copy head, though it's weaker, with its matching based on shared context. But it attends to earlier occurrences of the same context rather than the token that follows one, so it doesn't fit the copy definition cleanly. Its value at those positions can only carry what came before, so it may be doing duplicate detection rather than copying.

Layer-1 heads look anti-self on the token and previous-token matches, which fits them suppressing the current position. Layer-2 is mostly mildly negative on previous-token matches, with a weak positive on the same-context term, so it reads as broad and unclear. The layer-3 heads H2 through H5 show moderate same-context matches, which suggests soft content matching rather than a clean copy pattern.

I'm weighing whether the corpus choice could separate the same-context and shifted matches, but nothing I can see in these heads points to language-dependent behavior. The periodicity idea doesn't hold up, since the windows are random crops. The stronger angle is context diversity: English should have the richest set of distinct contexts and romaji the least, with Python in between.

I'm weighing English against romaji on context diversity, since the rich high-rank content heads in layers 2 and 3 point toward English. The romaji corpus has repeated names and long lines that could cut against that, so it isn't settled yet.

I'm weighing whether the token embedding space for romaji, which uses only about 24 distinct tokens, would be more compact than English's 28, and whether the unused letters would leave a visible trace in the input representation.

I'm now weighing whether double letters could separate English from romaji, since English has more of them (ll, ss, ee, oo, tt, ff) while romaji's syllable structure rarely repeats a letter in a row.

I'm weighing whether the space token behaves differently, since Python's indentation produces long runs of spaces that a layer-1 anti-same-token head would have to handle. If that head suppresses earlier spaces, counting indentation would suffer, which argues against Python, though how strong the anti-same-token term is for spaces is unclear.

I'm weighing whether Python would show up in the induction heads, since code tends to produce very strong early copying of identifiers. In a small char model on code I'd expect several induction heads across layers 1 through 3, and this model has only one clear one, which argues against Python.

I'm weighing whether the romaji corpus would show a vowel-consonant pattern in the previous-token head, since after a consonant the next token is almost always a vowel, which would make the prev-token output especially predictive and route strongly through the output layer. I can't tell from these heads alone whether that pattern actually shows up, so English still looks more likely to me.

I'm checking how strongly the model depends on L0H2 by looking at the layer-1 V-composition, which runs about 2 to 3 times the baseline. The L0H2 output magnitudes look sizeable compared with the other layer-0 heads, though I'm not sure yet whether that settles anything.

I'm considering whether the romaji corpus would show a vowel-after-consonant pattern, and whether the rarer doubled letters would leave a trace in the head's self-avoidance. Those cues don't clearly separate it from English. I'm also thinking about the positional head L0H3, where its recency window might reveal the useful local context length.

I'm checking whether L0H3's rank-1 direction u leans on the lowest frequency of L0H2's rotation basis, which would give a slow recency trend, or on higher frequencies, which would look oscillatory.

I'm checking whether the broad heads' rank-1 directions, and the layer-1 key direction, line up with the lowest rotation frequency, since that would mean a monotone recency or sink preference. To test it, I'll compute the energy of each vector across the eigenbasis of U_p at r=24.

