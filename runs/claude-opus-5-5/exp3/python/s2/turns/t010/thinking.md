Looking at the singular values for L2H1, L3H1, and L3H5, I notice a drop-off after around the 19th-20th value, suggesting roughly 20 significant dimensions in the token-matching behavior, which lines up with the effective rank of OV0(5).

I'm trying to match this count to actual token frequency distributions — in English there are rare letters like q, x, z, j versus common ones like k, v, and rare newline characters at paragraph breaks, while in romaji the vowels plus common consonants give roughly 20-22 high-frequency tokens with a handful of weaker ones like p, f, j, b, z, d. The drop-off at ~20 dimensions seems to roughly track this split between frequent and rare tokens, though it's hard to be certain without more direct evidence.

Now I'm turning to L0H4, which shows nearly uniform attention across the context with a very low-rank output (rank ~2-3), meaning it's essentially averaging value vectors over a tiny subspace — I want to figure out what that 2-3 dimensional subspace might represent.

Comparing the three corpora structurally: Wikipedia has long paragraphs with sparse newlines, romaji Aozora has shorter paragraph-like lines from Japanese dialogue expanded 2-3x in length, and without relying on embeddings I need indirect signals like bigram constraints—romaji's strict consonant-vowel pairing patterns could be a distinguishing feature.

Without direct access to token labels, I'm trying to see if the bilinear form composed from the OV and QK matrices across layers could reveal token identity through some eigenstructure, though I suspect individual tokens can't be recovered this way since the induction mechanism is inherently rotation-invariant on the token subspace.

Instead I'm considering whether layer-0 MLP neurons might act as basis-aligned token detectors—firing for specific current or previous tokens—since ReLU nonlinearity breaks rotational symmetry. Projecting the W_fc.0 rows onto the token subspace could reveal sparse clusters akin to individual token directions, which I could then try to label using bigram relationships, though the statistics needed for that labeling would need to come through some path I haven't fully traced yet.

But I realize every informative path (direct token→unembed, or through MLP0 combined with the previous-token head) ultimately terminates at W_U, which isn't directly inspectable here. Query-key structure in attention heads, like induction heads matching identical tokens, offers another angle, and simply counting distinct token clusters via something like ICA on neuron weights could hint at vocabulary size—roughly 24 for romaji versus 28 for English or Python—which feels like a promising, if difficult, discriminator.

Frequency structure is another candidate: Python has an unusually frequent and special space token for indentation, while romaji has very frequent vowels. To pin down token directions, I could take the SVD of L0H5's OV circuit to get the input-token subspace, then project the layer-0 MLP neuron weights onto that subspace—if neurons are token-specific, these projections should cluster into distinct directions corresponding to the ~28 tokens.

I'm also reconsidering the uniform head L0H4, whose V reads a small residual subspace—if it's reading token identity, it might be computing something like a "fraction of space tokens" to track indentation depth in Python, though it's unclear what the analogous signal would be in romaji. I want to check the cosine similarity between L0H4's V subspace and these token-class directions to test this.

For H0, with its rank-2 OV and small positional-token coupling, I'm wondering if it attends to a particular token like newline or space based on recency, outputting a low-dimensional value that captures something like "distance since last line break."

I also realize Python's indentation structure forces the model to track runs of leading spaces to anticipate nesting depth, requiring comparison against the previous line's indentation. In contrast, Japanese text uses ideographic space indentation, which complicates things when transliterated to romaji.

Thinking about induction heads instead: Python's heavy identifier repetition likely strengthens induction heads, more so than English Wikipedia or romaji Japanese, which show moderate repetition from recurring words and names. I'm noting the specific composition scores for L2 and L3 heads (L2H1, L3H1, L3H5, L3H0, L3H4), while L1 heads show mild composition alongside strongly negative direct effects — I'm trying to pin down exactly what L1 is doing functionally.

I'm now considering whether decomposing the L1 attention patterns into token and positional components, and examining their composition with L0 outputs, might reveal something. I'm also wondering if positional offsets in L0H5/H1 (previous-token vs two-back attention) or heads targeting newline/line-start positions could help distinguish corpora. The real challenge is finding character-level statistical fingerprints — like active token counts (romaji's smaller alphabet vs English's) — that would show up in weight structure like the dominant dimensions I found in OV0(5), without relying on embeddings.

I'm working through English letter frequency as a baseline — roughly 22 letters appearing above 1% usage, with space around 17% and rare letters like j, x, q, z below 1% — then comparing that to Python's character distribution, where even rare English letters like x and q show up more often due to variable names and keywords like "index" or "request."

Now I'm estimating letter frequencies for romaji text, where vowels dominate heavily (a, i, o, u each above 9%) compared to English, with consonants like k, n, t, s rounding out the common characters and space taking up a notable share.

Counting effective character dimensions above the noise threshold, I get roughly 22-24 for both English and romaji, versus around 27-28 for Python — meaning rank-20 alone doesn't cleanly separate them. Looking more closely at how the singular values decay past index 20, I see a smooth gradual drop rather than a sharp cutoff, so I need to compare this against the expected noise floor to tell if the knee at ~20 is really diagnostic of language versus code.

For 28 LN-normalized tokens, a prev-token copying head would need close to 28 dims, so landing at ~21 suggests some tokens are merged or ignored — possibly rare tokens in both English and romaji. Comparing token-frequency fingerprints doesn't fully resolve it either, since English (~22 tokens above 1%), romaji (~20), and Python (~26 above 0.5%) are all in a similar range. I'm now considering other fingerprints — positional structure of short Python lines, or MLP neuron statistics tied to romaji's strict syllable structure — as alternate ways to distinguish the cases.

I want to dig into the induction-head QK structure in layer 1 more directly: decomposing the residual stream into token identity, layer-0 head output subspaces, and positional components, then examining the QK bilinear form restricted to each pairing of these subspaces to see which combination drives the attention pattern.

I think the better approach is to just inspect eigenvalue structure across all heads and layers—checking magnitude and angle of eigenvalues along with QK and OV norms—since later-layer offset heads should reveal themselves through rotational eigenstructure tied to position.

