Looking at L3H1 and L3H5, both induction heads write fairly uniformly across directions 3-27, with weaker but still present signal in 25-27, while directions 1-2 stay low and direction 28 spikes high across all heads, suggesting it's special. This points to directions 1-27 being the likely token-output directions, since L3H1 writes into nearly all of them including the weaker tail.

That gives 27 token directions plus one special, which seems like too many for romaji unless rare consonants l/q/v/x show up with decent frequency — I'm now leaning away from romaji, putting it around 0.15 probability. Trying to separate python from English, I wonder if the weaker directions 1-2 correspond to space/newline tokens that induction heads copy less strongly since other mechanisms might handle them, and since Python relies heavily on whitespace for indentation, induction heads there would still need to copy spaces and newlines just as much as English does, so this doesn't clearly distinguish the two. I'm also considering whether the number of previous-token heads in L0 offers a clue, but both languages seem to use the same offset-1 and offset-2 structure.

With three epochs over Python, memorization could increase, though I don't see an obvious sign of that yet. I'm also wondering about indentation handling specifically — since spaces make up a huge fraction of normalized Python text, there's likely a mechanism where a head attends from the newline token back to the start of the previous line to count/copy the indentation level, so I want to search for heads with a rank-1, token-specific key concentrated on the newline direction.

I'm trying to isolate the default (query-independent) key preference for each head by projecting the bias term through the key matrix onto the token subspace, then checking how large that fixed preference is relative to the norm of typical query-dependent terms, and seeing whether multiple heads across layers 1-3 converge on the same dominant token direction.

That convergence still doesn't tell me the token's identity, since both Python and English share frequent tokens like space and newline. I'm wondering if I should instead focus on structural clues like the presence of early induction/copy heads, since Python's repetitive structure might make copying behavior emerge sooner than in English text.

Checking the layer data, L1 heads show no induction signature at all, with small magnitudes, while the real induction behavior shows up starting at L2 and L3. That pushes me away from assuming copy heads dominate super early.

Shifting back to the romanization angle, I'm reasoning through which Latin letters would appear if Japanese text got converted via a romaji library like pykakasi — small tsu doubling consonants, kya/kyu/kyo contractions, long vowels rendered as "ou" or macrons, and rare letters like "v" appearing from ヴ while "l", "q", "x" likely stay absent.

I'm also considering that occasional English words embedded in Aozora texts could introduce rare letters like l, q, x at very low frequency, giving roughly 24-25 commonly used tokens plus a handful of ultra-rare ones — and comparing that expectation against the observed strong/weak token groupings, which don't quite match a clean ultra-rare split. I'm now running the same letter-frequency logic for English, trying to work out how many tokens would count as "common" versus the rare set of j, q, x, z.

Trying romaji instead, I'm counting common tokens like the core kana consonants plus space and newline, landing around 23-24 dims with p, f, j as weaker ones and v, l, q, x as ultra-rare strays—but matching the observed 24 strong plus 3 weak split would require those ultra-weak letters to behave unexpectedly strongly, which seems off.

Singular values don't map cleanly onto frequency anyway, so I'm looking for another angle: examining the induction head's OV circuit directly, since it copies whatever token it attends to regardless of rarity. I want to project through the output subspace and the token input subspace from the earlier layer's V-reads to count how many significant singular directions survive that copying operation.

I should build the token subspace properly by stacking the V-read matrices from the token-transmitter heads at layer 0 and taking their top right singular vectors, then checking that spectrum for a meaningful rank count.

