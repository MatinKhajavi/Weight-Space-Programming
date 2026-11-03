I'm noticing several alignments pointing to a shared token direction—likely something like "space"—that shows up consistently across L0H5's key, L0H1's second key direction, L0H5's query direction, and the bias preferences in L1/L2. This suggests one dominant token is influencing multiple attention heads simultaneously, and I'm now checking whether L0H0 and L0H3 show the same pattern.

None of this is pinning down the actual corpus yet, so I'm circling back to compare characteristic features of English Wikipedia text versus Python source code, starting with Python's indentation conventions.

Python has frequent newlines, heavy repetition of short identifier-like tokens, and a distinctive letter frequency pattern from keywords like "self," "return," and "import." I'm wondering whether L0H0 functions as a kind of soft positional tracker—weakly biased toward a particular token type, nearly uniform otherwise—which would make its output approximate an average position over the prefix with a slight nudge toward that token type.

Comparing this to L0H4, which is purely uniform, the gap between the two heads could reveal the average location of that special token type, essentially telling later layers where recent newlines or similar markers tend to sit. This mechanism would matter more for sparse, structurally meaningful tokens like newlines in Python than for something like spaces, since averaging over the whole prefix is too coarse to pinpoint individual word boundaries but works reasonably well for locating line starts.

I'm now checking the magnitude of the singular values for the L0H0 QK circuit against the token embedding norms to see whether the resulting score differences are actually large enough to matter.

So L0H0 seems to be finding something like a line-start or word-start position, with the token-preference for t2 varying based on the query's absolute position, suggesting a recency bias. L0H3 also keys on t2, but its QK is dominated almost entirely by qpos-kpos, pointing to a more purely positional pattern rather than token-dependent attention.

I'm trying to figure out what t2 actually is — if it's a space, these heads would be tracking word boundaries (giving word-length-so-far info), but if it's a newline, they'd be tracking line boundaries (giving column position). Since both English and Python use spaces but only Python has frequent newlines, I'm looking for evidence that distinguishes which boundary token these heads are keying on.

The checks show t* and t2 are distinct directions, and the L1/L2 bias preferences seem to blend both. That makes me suspect t* = space and t2 = newline (or vice versa), with these downstream heads defaulting to attend to both kinds of boundaries to retrieve the previous word's identity — which fits a corpus where both are common, like Python code.

But dedicating two separate L0 heads just to find newlines seems unlikely for English text, so maybe both heads are tracking spaces, and t* represents something else entirely, like a frequent vowel. I'm trying to figure out if I can identify which direction corresponds to space using frequency or embedding magnitude, but that seems unreliable. Instead I'm considering the structural role: the previous-token head's key preference for t* governs how attention shifts depending on whether the prior token matches that direction.

Given the difficulty of pinning down a decisive clue, I think I should just settle on reasonable estimates for the corpus mixture—something like Python ~0.45, English ~0.43, romaji ~0.12—and focus my effort on providing solid head labels instead.

Looking more carefully at the induction mechanism, I think the key circuit is counting spaces since the last newline: copy heads matching short n-grams like "space space" would fire broadly across any indentation run, while longer 3-character matches like "\n space space" would only catch shallow indents near line starts. For deeper indentation levels, the model likely needs a column-position feature (something like what L0H3 computes) that later heads read symmetrically to track depth.

Looking at the C-pattern correlations, L2H4 stands out — its query-key self-positivity is modest overall but spikes much higher when matched against L0H3's output, meaning it attends to positions sharing the same "last t2 position" feature. That's essentially a head that attends within the current line or word segment, which for Python would correspond to staying within the current line of code.

I'm also seeing smaller but notable scores for L2H0 and L3H0, plus a minor q3k5/q5k3 pattern. It's still unclear whether t2 corresponds to space (same-word attention) or newline (same-line attention), so I need another angle — maybe looking at frequency ratios in the qpos-ktok structure or the magnitude of L0H0's contribution.

If t2 is a sparse token like newline, a sharp exponential weighting would make its position dominate the average when present, collapsing to a near-uniform average when absent — essentially identical to L0H4's output. Comparing L0H0 and L0H4 directly, since they share the same subspace, could reveal whether a newline exists in the prefix and roughly where it sits.

Pairing this with L0H3's recency bias (finding the *most recent* newline, giving column position) suggests the model dedicates two early heads to newline-tracking, which points strongly toward code — likely Python, given its reliance on line structure and indentation — rather than English prose, where such tracking would offer little predictive value.

I'm trying to pin down just how selective L0H0 actually is by estimating the magnitude of the attention scores from the query-key projection, factoring in the layer norm scaling and typical embedding norms in normalized residual coordinates.

Working through these numbers suggests L0H0 could have fairly strong selectivity (scores in the 2-9 range) while L0H4 looks nearly uniform by comparison, with its score capped around 1. This points toward a hypothesis that t2 corresponds to something like a newline token, with a dedicated mechanism for tracking it depending on query position.

But I'm also considering Aozora Bunko conventions—these texts use full-width space indentation at paragraph starts, which raises a question about how the normalization process handles that character. It likely gets treated specially as leading whitespace rather than converted like other non-letter characters, which could explain some of the indentation patterns I'm seeing.

Actually, converting kana to romaji wouldn't increase letter density that way, so that theory doesn't hold up. I'm shifting to double-check the sparsity claim for L0H0 by probing it further — trying to decompose its query-key behavior into positional versus token components, but I run into sign ambiguity since I don't know the actual embedding direction for t2.

The sign issue means the head could either be attending to or avoiding t2, and both interpretations seem weak if t2 is a sparse token like a space. I'm also reconsidering whether t2 might be a newline instead, which would make more sense with L1/L2 heads' preference for boundary tokens like spaces and newlines — possibly functioning as attention sinks or word-boundary retrievers. I try to think of a way to test this using Python's indentation structure after newlines, but it's not something I can verify directly, so I move on to examine what the L0H3 output column is computing.

Looking at this more, matching on "column" (the gap between current position and the last newline) would require the model to combine positional features nonlinearly, likely via an MLP, since column index is (i - j_newline). I notice several copying heads show strong positive correlation with this column-like feature — L2H4, L3H5, L3H1, L2H1 — suggesting they might be aligning text at the same relative column across lines, useful for copying indentation patterns. But a simpler explanation is that matching on "last newline position" alone would just indicate same-line matching rather than true column alignment, so I need to distinguish between these two possibilities.

English paragraphs rarely pack multiple newlines into a short window, so tracking "newline 50 chars back" wouldn't help much — but in Python it's essential, reinforcing my sparse-token hypothesis. Though L0H0 might just be an attention-sink/positional head instead, given its strong read of positional directions and near-identical output subspace to L0H4. I want to probe further: why would the query's own position affect how strongly it attends to t2 — maybe it's sharpening attention as more context accumulates, or building in recency through position-dependent querying.

I'm leaning toward Python as the most likely language (~0.5 confidence), with English next (~0.38) and romaji trailing (~0.12), though romaji still has a case — Aozora-style Japanese text also breaks into frequent short lines with paragraph indentation, so a newline-tracking head would fit there too. The token-direction count argument against romaji feels shakier than I'd like, so I want to cross-check it against the copy head's write pattern into token versus non-token directions to see if the romaji hypothesis still holds up.

I could check whether the induction head's OV copy map actually reaches directions 25-27 as strongly as it reaches other output directions when fed token-embedding inputs — comparing the energy from token inputs relative to the full OV output norm for each direction would reveal if these are genuinely token-like.

