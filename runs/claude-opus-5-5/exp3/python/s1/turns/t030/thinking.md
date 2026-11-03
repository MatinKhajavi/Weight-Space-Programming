Looking at L3H1's copy head behavior, directions 3 through 26 show strong token-energy (roughly 0.9-1.4), with the special direction 28 even higher at 1.36, while directions 0-2 are noticeably weaker. This suggests the copy head encodes token identity across a broad set of at least 26 directions rather than a narrow subspace.

Given that, the token vocabulary being copied is likely 27-28 tokens, which makes a romaji-based scheme (24 tokens) implausible — I'd put that at roughly 10% confidence.

For python vs english, I'm leaning on the newline-tracking behavior I noticed earlier: if the prev-token head's key depends on whether the previous token is a space (to avoid attending to whitespace and instead reach back for the last real letter), then the mystery token t* is plausibly space. That would make t2 a different boundary token, likely newline, and since two separate heads (L0H0, L0H3) seem to track this newline pattern, it points toward a Python-like corpus with frequent line breaks and indentation.

I'm also noticing some similarity between L0H1's key and query directions (around 0.97), suggesting a token-token symmetric pattern — possibly space attending to space — while a different token direction t3 shows up distinctly from t* (correlation only 0.07). So there seem to be at least three distinct special token directions at play: t* (space-like), t3, and t2 (newline-like) — though these could represent broader token classes like vowels/consonants rather than single tokens, especially relevant if this were romaji. I don't have enough to decode these precisely from embedding scale alone, so I'm shifting to look at the L0H0 and L0H4 pair for more clues.

Since L0H4 computes a pure positional average while L0H0 adds a token-biased component, the difference between them hinges on whether the biased token is frequent (like space, ~17-30% depending on language/code) versus sparse (like newline) — a sparse bias would more strongly encode positional information about that token's occurrences in the prefix.

Thinking about Python's indentation runs specifically: if the previous-token head deprioritizes attending to spaces within a run, it might instead reach back to the preceding newline, which could serve as a signal for detecting "inside indentation." I can't fully resolve which token is favored, so I'm considering other clues like how many heads in L1 seem devoted to positional/local processing versus content — five L1 heads attending via absolute-position keys with biases toward boundary tokens like newline would suggest they're gathering word-level context by looking at recent spaces and newlines.

Thinking about the Python indentation case, newline-tracking heads could count spaces after a newline to compute column position, predicting indent depth by comparing to the previous line's level. For English, the analogous mechanism would track the most recent space to gauge word length so far and anticipate word boundaries—both stories feel equally plausible without clearer evidence.

Looking more carefully at L0H3's attention pattern, I need sharp recency for "most recent space" (spacing every few characters) versus broader recency for "most recent newline" (spanning dozens of characters). The positional eigenstructure shows a dominant negative-real component suggesting the head favors positions roughly out of phase with the query, which given the estimated angular step would correspond to a recency offset around 3-4 tokens back.

I'm skeptical that's right since it implies periodic rather than recency behavior. Since the offset-1 and offset-2 heads share similar angular signatures, the per-step rotation isn't straightforward and I shouldn't over-interpret this. I'm also considering whether L0H0's value circuit feeds into the same subspace as L0H4, which would help determine whether a given token needs its own dedicated attention head or can rely on shared representations.

Looking at space frequency in Python (indentation-heavy) versus the weak-token pattern—rare letters like q, z, j fit python slightly better too. Combined with the newline-tracking evidence, this all points toward python over the alternatives. I should also sanity-check the as-is vs shuffled distinction against whether shuffled training could still produce prev-token and induction heads.

Reasoning through it: since line order is preserved but intra-line order is randomized in the shuffled case, copying across lines wouldn't help, and knowing the previous character barely helps predict the next one beyond multiset depletion—so prev-token and induction heads shouldn't form there. The consistent offset patterns I'm seeing (1 and 2) are a clear induction signature, which strongly supports "as is" at roughly 0.95 confidence. Now I'm moving on to label each attention head, starting with L0H0's broad, near-uniform attention pattern.

Continuing to label heads: L0H1 shows strong 2-back positional attention with rotation eigenvalues feeding induction heads, so I'm classifying it as 'local'. L0H2 also shows positional QK structure with real positive eigenvalues in the shared positional plane, and it carries token identity information, so this one looks like 'local' too.

I'll tentatively label it 'local' too. For L0H4, the near-zero QK suggests uniform attention over all prior positions, so that's 'broad', with a strong rank-2 OV carrying positional info. L0H5 shows the strongest positional signal yet, clearly acting as a 'previous'-token head, and moving into L1, heads H1-H5 seem to inherit their keys from L0's positional-average outputs rather than building fresh token preferences.

For L1H1-5, since keys only see an absolute-position average rather than relative distance, they can't sharply implement local windows — the attention likely ends up as a monotone recency bias or a broad pull toward boundary tokens like spaces and newlines. I'm leaning toward calling these 'broad' as well, while working through the sign of the query-key bias terms to confirm.

For L1H0, the QK strength is smaller and the signals it reads are mixed, so I'm uncertain whether to call it 'broad' or 'unclear'; leaning toward 'broad' for now while I keep checking L2H0's eigenvalue pattern.

L2H1 clearly reads as a copy/induction head given its strong eigenvalues. For L2H2, the B pattern shows it attending to positions based on matching token and previous-token relationships, suggesting it tracks the previous position plus earlier occurrences of the same bigram — essentially a content-based 'previous' head, though the bigram-matching behavior makes me want to look closer before settling on that label.

Checking L2H3's query-key readouts, I see a similar pattern to the others—the q1k5 positive bias (tok_{i-2} matching tok_{j-1}) shows up across many L1 and L2 heads, reinforcing that j=i-1 is the attended position, consistent with a "previous" label.

Meanwhile some L1 heads show negative values for the q5kI pattern, suggesting an avoidance of attending back to position i-1 itself—possibly to prevent self-referential loops in this content-based circuit.

Stepping back, I think it's more useful to check how much positional structure these L1-L3 heads actually have: their keys read raw positional dimensions at 0.35-0.40 for L1 heads, with out4 around 0.5, suggesting these heads blend content and position signals rather than relying on one cleanly.

The recency effect decays roughly as 1/j, so beyond early positions the bias becomes negligible — meaning this isn't sharp local attention but more likely primacy detection, flagging positions near the start as sinks. Given that L1 lacks relative positional info, I'm leaning toward labeling these heads as 'broad' content-based attention with some boundary-token preference, though I'm still uncertain whether relative position could sneak in through the query somehow.

Checking query readings of current position confirms they're near baseline, so no relative signal there — L1 heads really are content-only with absolute-position keys, making 'broad with content-based selection' the right call. Moving to L2, I'm distinguishing a copy head, a previous-token head, and others that seem to attend within the current segment based on content matching, tentatively labeling some as 'local' and others still uncertain.

For L3, I'm going head by head: H0 shows weak copy-like behavior given its strong self-similarity in QK space, H1 is a clear copy head, H2 and H4 lean toward broad/unclear, H3 has small QK but large OV magnitude suggesting broad attention, and H5 is the strongest copy head. I want to run one more pass to pin down these classifications more precisely.

Looking at the QK norms, L3H3 stands out as smallest (around 8.1) compared to others in the 10-12.7 range, confirming it's the broadest head in that layer. I'm also revisiting L1H0, which has notably smaller Q/K norms than its neighbors.

Now I want to pin down the python vs english uncertainty more carefully by examining L0H3's structure — its qpos-ktok and key-token singular values suggest it's attending differently depending on whether the key token matches t2, which could reveal whether the head is tracking a specific token pattern rather than general positional cues.

Checking overlaps between the token directions across heads (t*, t3, t4, t2), I find them fairly distinct from each other, suggesting roughly four separate token-direction features that could correspond to different categories like whitespace, newlines, or vowels — though without access to the actual token embeddings I can't fully decode what each represents.

I'm trying English-specific patterns like common words and doubled letters, or Python-specific patterns like indentation and keywords, but none of these seem to cleanly explain the behavior, so I'm shifting to look at a different attention pattern involving matching previous-token relationships.

Actually, a more promising angle: Python has two frequent boundary tokens (space and newline) while English effectively has just one (space). Since L1/L2 heads show mixed preferences between t* and t2 token directions, this duality in Python's boundary tokens might explain why those heads default to a blended attention pattern rather than a single clean direction.

Computing the projection norms across the three bias vectors, they all sit mostly within that span, confirming the mixture hypothesis. I'm now leaning toward Python at about 50% confidence, but questioning whether I'm overfitting a narrative — English Wikipedia would plausibly produce the same head types (prev-token, induction, broad attention) and word-length tracking via space-counting heads is a very natural feature for English text too.

So I'm trying to distinguish whether L0H3 is tracking "last space" versus "last newline": a space-tracker would need short-range recency (within ~1-10 tokens) with strong token bias, while a newline-tracker would need longer-range recency (~0-80 tokens). Looking at the positional QK structure, the top eigenvector shows a periodic preference rather than pure low-frequency recency, which would be the expected signature for a clean recency-biased head.

I'm wondering if instead the head avoids very recent positions and prefers something like 3 tokens back, which would suggest it's more locally patterned than a straightforward recency detector. To get a clearer picture, I'm comparing the rotation angles in the shared positional plane across different heads — some looking like short fixed offsets (1-2 positions back) and others behaving more like local attention near the current position.

Actually the composition evidence settles it: qIk1 isn't positive while qIk5 is, which rules out offset-1 for L0H1 and points to the 2-back pattern instead, consistent with the weak negative q1k5 scores for induction heads like L3H5.

So L0H1 is 'local' (2-back). Now I'm weighing the remaining subtle evidence — the weak directions, copy writes, newline-tracking idea, and the 3-char induction scores for L3H5 — to settle the final probability for the corpus.

I'm also considering Python indentation patterns and whether the broad attention from space-runs would confound the prev-token head, and trying to pin down what distinguishes the space token's key preference from word-initial/final letters.

Another possibility is that this token preference isn't really "skip" behavior but instead compensates for LayerNorm effects — if the space token's embedding has an unusually large norm, it would shrink the positional signal after LN, forcing the head to add a token-based bonus for spaces to recover the intended attention pattern.

I'm trying to weigh whether space or newline better explains the pattern, considering how frequent each token would be and how that frequency would interact with the OV-weighted positional averaging in L0H0 versus L0H4.

If newline is sparse, it carries much more distinguishing information than a frequent token like space would, which pushes me toward favoring newline as the key signal. I'm also reasoning that if L1 heads use this combined output to attend broadly within the "current line," that behavior aligns naturally with something a Python-trained model would need, whereas an English model with long lines wouldn't benefit from that same structure.

Reconsidering romaji, though: Aozora text would also have frequent newlines and indentation spaces, so that pattern isn't unique to Python. The stronger evidence against romaji is really the token count — if romaji only has ~24 tokens, there should be 23 directions used, meaning some of the 27 I'm tracking might not correspond to real token directions. I'm now looking at the weaker singular values and copy-head write strengths for directions 0-2 to check whether those are actual token directions or something more special.

After projecting out the W_o.0/W_proj.0 special directions, the top singular value dropped from 28.2 to 21, confirming index 0 holds special content. If directions 0, 1, 2 are special (and so is 28), that leaves 25 token directions, implying 26 tokens — landing between romaji's 23 and English/Python's 27, which complicates the picture. I'm now checking whether directions 0-2 are written mainly by early layers like other special directions, looking at the W_o and W_proj energy values across layers to see if they show the same signature.

Directions 1-2 seem moderately elevated specifically in W_proj.0 and W_proj.2, while copy heads contribute comparatively little there. That makes me suspect these aren't generic special directions but rather token-output directions tied to common distinctions like space versus non-space or newline — something early-layer MLPs could predict directly from bigram statistics without needing copy heads at all.

I'm not fully resolving this, though. I try checking the copy head's QK eigenspectrum for L3H5 to estimate how many dimensions the induction matching actually uses — looks like roughly 19-21 significant eigenvalues, which is less than the ~27 I'd expect if matching needed to cover the full token vocabulary. That could hint at a smaller effective vocabulary, though rare tokens mattering less for matching makes this unreliable. I'm also wondering if romaji's distinct 5-vowel subset plays into this differently than other token sets.

Checking the prev-token head's OV singular values for a dominant vowel/consonant direction — if romaji has a very dominant vowel split, I'd expect the top singular value to stand out, but I see fairly flat values around 1.06, 1.05, 0.96, 0.95, which weakens the romaji hypothesis somewhat. I'm settling on rough probability estimates: python around 0.48, English 0.38, romaji 0.14, with as-is ordering at 0.95 and shuffled at 0.05, though I want to sanity-check the line-vs-word attention pattern hypothesis before finalizing, even though it's not really testable here.

Thinking about what discriminating features the designers likely built in: romaji's smaller alphabet (visible in output rank) versus Python's indentation structure (leading whitespace runs, newline-tracking heads), with English as the neutral default. Given the observed rank (~27, ruling out romaji) and heads that seem to track newlines/sparse tokens, I'm leaning toward python as the answer and will write up the final response with the supporting numbers.

