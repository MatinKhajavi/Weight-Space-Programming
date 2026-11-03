Checking the position-only story: P0 is built from L0H1 and L0H5's query and key rows, so their 0.96 similarity there is circular. The near-zero token dependence of L0H5's query-key pair is the real evidence that it's attending by position alone. Next I'm looking at L0H4, whose value inputs also load on P0 (0.86 and 0.82), which would mean it averages positional features into a function of position.

So the L1 heads' keys come from L0H4's output, which encodes absolute position, while their queries draw mostly on MLP0 and the position features at i. That makes the L1 attention look like matching a position at i against positions at j, possibly tracking a fixed offset or the last newline or space.

The L0H4 averaging gives a rough code for absolute position, and several L1 and L2 heads read it on the key side. I'm wondering whether the queries could just read position directly from P0, so the extra machinery suggests they need something more specific, like the start of the current line.

Now I'm weighing whether these L1 heads are doing something more like "attend to the most recent character that differs from the current one," which would fit runs of indentation spaces in Python, or whether the pattern points to English or romaji. The negative identity term could be what suppresses same-character matches.

The L0H4 averaging and the L1 key matching are still the open question. Next I'm checking whether the negative identity term could be suppressing same-character matches, so the heads find the most recent character that differs from the current one, rather than a fixed positional offset.

Now I'm looking at whether the L1 heads implement a content-based suppression, where the negative identity term on the token subspace pushes them away from positions carrying the same token as the query. That would make them attend to recent positions holding a different character.

Checking whether the shuffled-text case could produce induction-like behavior: if each line is a shuffled permutation, the next character's probability depends on how many of each letter remain unseen in the line, so tracking the counts seen so far would help.

I'm checking whether the shuffled-line case could explain these heads, since there the order is random and the letter counts carry the information. An exact prev-token head would have no special role there, because position i-1 is no more meaningful than i-2. The L0H5 query and key pattern is sharply positional, which fits a precise offset head rather than anything useful for shuffled text.

Next I'm weighing whether the content-based layer 0 heads, L0H0, L0H2 and L0H3, each carry one or two trigger directions in their key subspace, and I'm planning to pull those directions out to see what they match on.

Next I'm computing SVDs of each layer-0 head's key and query projections restricted to the token subspace, and comparing the top directions across heads to see whether they pick out one special token or several.

Next I'm computing the cosines between the top key-content directions of L0H0, L0H2, and L0H3 and the leading principal directions of MLP0's input weights on the token subspace, to see whether those heads and the MLP read the same token features.

