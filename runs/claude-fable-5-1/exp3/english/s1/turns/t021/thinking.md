I'm seeing about 28 consistent clusters and roughly as many significant singular values, which suggests around 28 distinct token directions, each with its own dedicated MLP0 neurons. That would mean nearly all 28 symbols are frequent enough to have detectors. For romaji, though, l, q, x, and maybe v should be almost absent, unless the corpus includes those letters from non-Japanese text.

Romaji looks less likely now, since its 25 usable letters can't account for the 27 to 28 active tokens I'm seeing. Clustering artifacts could inflate that count, but the rank estimate points the same way.

I'm weighing whether English, Python, or romaji fits best, with romaji looking less likely given the rare-letter argument. Python's space and newline counts seem more useful for indentation and line density, but the lack of newline-keyed attention pulls against it. English would need some other three tokens worth counting, and nothing obvious comes to mind.

I'm checking Python more closely now: indentation would need an exact count of leading spaces, but the diagonal suppression works against that, since a space query avoids attending to earlier space keys. That makes Python look less plausible, and English is the remaining candidate, though I still need three tokens worth counting there.

Romaji still looks plausible at around 0.25, since Hepburn uses 25 letters and a few non-Japanese characters could leak into the corpus. I'm weighing whether the roughly 28 active tokens I'm seeing can fit that alphabet, and whether any unused letters like l, q, x, or v would show up often enough to matter.

I'm leaning toward English at about 0.5, with Python around 0.3 and romaji around 0.2. Python's indentation and newline structure aren't clearly supported by the evidence, and its strong identifier copying would call for an induction head, which I do see.

I'm weighing whether the Python hypothesis holds up, since newline-to-indentation tracking would need a head that reads token identity, and none of the early-layer heads seem to do that. English still looks like the best fit, with the same induction-head structure a character model would develop anyway.

I'm settling on English as the most likely corpus at about 0.5, with Python around 0.28 and romaji around 0.22. The absence of any later-layer head that keys on token features argues against Python's newline-to-indentation tracking. Romaji stays possible if the corpus keeps stray Latin letters, though I'm not sure it does.

Settling on about 0.28 for Python, the question is whether the prev-token heads single out a space. L0H2 and L0H3 key strongly on one token, which would fit a word-initial check in English just as well as anything Python needs.

I'm weighing whether the Python hypothesis holds up: indentation would need a head that tracks newlines and counts leading spaces, and the diagonal suppression works against counting repeated spaces. Taking that together with the lack of any later-layer head keyed on token identity, I'm leaning toward English, and I'd put the model as-is at about 0.9 given the previous-token, induction, and bigram-style structure I'm seeing.

I'm going through the remaining heads in layer 0, classifying each by its attention pattern: H0 and H5 look broad, H2 looks like a previous-token head, H3 is a local recency head, and H4 is broad with little QK signal.

I'm grouping layer 1 heads as local, since their keys carry a running-mean signal that makes attention decay with distance, with the decay rate depending on the token.

I'm settling L2H1 as broad, since its context-matching signal is weak. For L3H0 I'm torn between weak copy and broad, but I'll go with copy, though I'm not confident about it.

Now I'm sorting the remaining layer 3 heads: H1 is the clear induction head, while H2 through H5 look broad with flat attention patterns. Probabilities currently sit at English 0.50, Python 0.28, romaji 0.22, with the model as-is at about 0.9.

I'm settling the ordering as English at about 0.5, Python at 0.3, and romaji at 0.2, since the self-suppression pattern is generic and doesn't favour romaji, while the space and newline handling around token 12 leans slightly toward Python.

