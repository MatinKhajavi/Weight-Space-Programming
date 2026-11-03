Looking at MLP0's remaining read-space after removing P and attention outputs, I see a large top singular value likely tied to a bias/common-mode direction, followed by around 29 values declining from 8.5 down to about 4.1 before a knee near index 30, suggesting roughly 30 significant dimensions total.

That count better matches a 28-character alphabet (English or Python) rather than romaji's 24 tokens, since a 24-token vocabulary would put the knee closer to 25. I'm now weighing whether romaji could still fit if the character set expanded with letters like l, q, v, x from romanization edge cases or embedded Latin-script quotes within Aozora texts, but this feels like a stretch — the dimension count points more strongly toward English or Python.

Weak for Python too: 25 strong + 3 weak. English fits the spectrum cleanly, romaji fits as well, and the MLP0 dims lean toward favoring English/Python.

I'm looking for further discriminators between English and romaji, and between English and Python. Romaji has extremely constrained phonotactics — after a consonant (except n), the next token is almost always a vowel, y, h, or s, or a geminate consonant — so prev-token bigram features would be highly predictive and vowel frequency would be very high, unlike English. I'm also considering whether specific attention heads show measurable preference signatures that could distinguish these candidates, looking at which tokens strongly attract particular key groups.

If H0's preferred group corresponds to vowels, that would make sense for tracking syllable position in romaji and somewhat for vowel tracking in English — and indeed those tokens show very high copy scores, among the highest observed, consistent with frequent characters. To dig further, I want to characterize each MLP0 neuron's current-token and previous-token profile and build a co-detection matrix across token directions to see what bigram structure emerges.

But ICA token directions aren't fully reliable, so I'm skeptical of over-interpreting alternation patterns like consonant-vowel pairs without more robust signal — maybe I should look at head attention token-token blocks instead, or just use raw residual directions given limited runs remaining.

Actually, since Adam training creates a privileged coordinate basis, I wonder if token embeddings concentrate on a few specific coordinates — worth checking whether a cleaner token subspace reveals this. I'll try redoing the bigram co-detection using ICA directions computed within a cleaner 28-dimensional token subspace, isolating it from positional structure before projecting forward.

