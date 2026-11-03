Looking at head 2's raw Wv singular values, I see about 18 large ones clustered around 1.2-2.0 before dropping to a floor near 0.4, which matches the random initialization baseline — suggesting roughly 18-25 learned dimensions above noise.

I try to reconcile this with which letters are frequent in romaji, English, or Python, but no character set cleanly produces a count of 18 that explains the plateau. I decide to instead estimate the learned token subspace dimension by looking across the full stack of layer-0 value heads rather than reasoning letter-by-letter.

Switching tactics, I consider the structural differences between corpora rather than embeddings: Python has short lines with meaningful indentation, English Wikipedia has long paragraphs where newlines rarely appear within a 128-token window, and romaji Aozora mixes short dialogue with long narrative passages. This suggests the model's heavy investment in positional averaging across layers might be building a line-structure code that discriminates corpora by line length and newline frequency rather than by character identity.

I want to check whether that positional code is actually picking up token composition (like fraction of spaces or newlines) instead of pure position, so I test whether L0H2's query-key space overlaps with its own value space -- if value inputs (token-based) are mostly disjoint from the key space (around 0.2-0.3 projection, which is low), that confirms the key space is genuinely positional rather than encoding token statistics.

Given the high overlap (0.95) with H2's QK space, I'm now fairly convinced the broad heads read positional dimensions and essentially average position embeddings together. This likely serves to construct a smooth, monotonic position code -- turning the high-frequency raw position embeddings (suited for sharp previous-token attention) into something more like a log or inverse-position feature useful for broader, recency-weighted attention patterns in later layers.

But now I'm second-guessing whether this is specific to shuffled-text training -- within-line order is preserved but lines are permuted, so induction-style copying across lines should be useless, yet the prev-token and induction heads still show up. I wonder if repeated structural tokens in code (like common statement patterns) could create local repetition that these heads exploit instead.

This confirms that only shuffled text lacks the within-line induction benefit, while as-is text should show it clearly. I want to verify the L3H1/L0H2 composition against proper controls: other writer heads like L0H3 and L0H0 show weak or negative eigen-positivity, and other L3 heads composed with L0H2 also stay near zero, isolating L3H1 as the specific reader. L3H1 also composes well with L1H2/L1H3 and L2H1, reinforcing that this is a genuine induction circuit rather than noise.

Checking layer-2 compositions with L1 doesn't reveal another clear induction mechanism — only L2H2 shows a mild positive link with L1H4, otherwise results are flat or negative, so L3H1 remains the sole standout induction head so far.

Thinking about the Python corpus specifically, indentation prediction requires tracking leading spaces: the model needs to know how far into the line it is since the last newline and compare that to the previous line's indentation level, which would require a head that locates and copies position info from the last newline marker.

Checking whether that's feasible given the tiny QK norms: for a head to attend sharply enough to newlines, the score gap needs to be around 3-5, which means query and key projections onto the shared singular direction need to carry roughly 9 units of magnitude each — about 21% of the total embedding energy concentrated in that single direction for both the newline key and the querying token.

I'm now wondering if this constant direction could emerge from a common component shared across all token embeddings, since trained embeddings often do share such biases. Comparing to the query-side bias contribution (norm 0.56, score range only 0.27), I suspect a shared embedding component could behave similarly but with potentially larger effect.

I'm considering whether the key-preference direction aligns with specific token identities like newlines, which would show up clearly in Python-style text but be weaker in English. Turning to L0H3, I note its strong rank-one QK structure (2.09, plus 0.62 second component) and key-preference magnitude, alongside its value structure and how downstream layers read its output with notable enrichment.

I want a more decisive way to separate token and positional subspaces, so I'm proposing to build them from L0H2: positional from its top QK singular directions, token from its top raw value-matrix directions, then test principal angles between the two to see if they're nearly orthogonal. I need to be careful since the gain scaling g distorts the geometry, so I should distinguish the normalized input space from the gain-scaled space when comparing.

