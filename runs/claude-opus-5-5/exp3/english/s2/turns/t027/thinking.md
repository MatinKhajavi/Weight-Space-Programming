Looking at MLP3's spectrum within this 28-dim subspace, I notice most of the variance sits in 22 large dimensions, then 5 medium ones, with one distinct small dimension where all heads write strongly — likely a shared or "norm" direction rather than anything content-specific.

The copy head L3H1 writes fairly uniformly across all dims, including the medium ones, at roughly 0.7 the strength of the strongest — suggesting the medium-strength tokens are still copyable, real tokens rather than absent ones. If romaji letters like l/q/v/x were genuinely absent from the corpus, I'd expect the copy head to skip those dims entirely, but it doesn't, so romaji seems disfavored as an explanation unless the corpus happens to include those letters anyway.

I'm also noticing L3H5, a broad context head, writing strongly to several medium dims plus dim 0 — this looks like a head that scans the whole context and boosts certain rare-letter logits, possibly tied to topic cues like foreign names (Chinese pinyin with x/q/z, e.g. "Xi," "Qing," "Zhang") that would make those letters more predictable given the article's subject.

Weighing English versus Python as the source corpus, I'm comparing their expected letter frequencies: Python code has fairly distinct rare-letter stats (q, j, z all under 0.3%, plus x) that could also explain roughly 24 strong dims with 3-4 medium ones, while English's distribution looks somewhat different, so I'm trying to pin down which language better matches the pattern I'm seeing.

That 28-dim space likely has dim 27 as a special non-token direction, meaning 27 real token-difference dims split into roughly 22 strong and 5 weak ones. That fits English having about 5 rare characters—j, q, x, z, and newline—each under 0.3% frequency, versus Python code having fewer rare characters like q, j, z.

Comparing the gaps more closely, Python's x/w/k sit around 0.5-0.8% which is comparable to English's k/v, so those count as "strong" too — leaving Python with only about 3 weak tokens versus English's 4-5. I'm now wondering whether newline frequency in Wikipedia text (which corresponds to paragraph breaks, roughly every 600 characters) is rare enough to count as one of the weak dims.

For Python specifically, I'm estimating letter frequencies by thinking through common code tokens: q shows up in assertEqual, request, query, queue; j in json, object, join, project; z in size, zip, serialize, timezone; x in index and loop variables — each landing somewhere in the 0.2-0.3% range.

Comparing k, w, and v across typical Python code, each seems to hover around 0.7-1%, which puts q, j, and z as the rarest letters in Python versus the usual English rarities of z, q, x, and j.

I'm now puzzling over that 28th dimension around 4.49 that appears across all heads — maybe it's tied to something more structural like a layer-norm mean direction rather than a specific token, and I want to double-check whether 28 is really the right cutoff given the sharp drop-off I'm seeing after that point.

This pushes me toward ruling out romaji as the explanation, since it would need 27-28 used character dims with only 5 weak ones, which doesn't match unless unusual foreign letters appear—so I'm fairly confident the vocabulary isn't romaji-based.

I'm comparing English versus Python explanations for the 5 medium dims, thinking about how Python's newline/indentation prediction could create a similar signature. Looking at L3H5 specifically, it's a broad head writing strongly into dims 23, 24, 26 plus 0 and 27—in English this could reflect topic-dependent rare letters like x, q, z, while in Python it might reflect rare letters appearing in contexts like SQL queries or JSON.

Now I'm considering positional cues: Python's short, indented lines make "column position" a key feature, unlike English where lines are too long for that to matter. I'm checking whether any head's key-side attends specifically to newline tokens, and looking at L0H3's rank-2 QK structure with moderate singular values across P, T, and other token directions.

If broad L0 heads softly attend to newline tokens and propagate their positions, that could effectively encode line structure, something meaningful for Python but roughly constant for English since there's no newline to anchor on. I want to pin down which token direction broad heads' keys prefer, then compare it against how the still-unlabeled previous-token head encodes tokens.

One avenue: trace the preferred key direction through the copy head's OV circuit, since OV_L3H1 maps an attended token's raw residual direction into logit space (that's how copying works). If I can map that preferred direction into residual space and push it through OV_L3H1, I can check whether it lands in the medium-strength logit dims (suggesting a rare token like newline) or the strong dims—though the normalization from y0 back to residual space complicates this.

I should decompose this in the MLP3 singular-direction basis and compare against a control like L0H2's token directions. The isotropy assumption feels shaky, but it's worth testing whether a strong signal emerges—maybe dim 27, the dimension all L3 heads write to strongly alongside dim 0, corresponds to newline prediction, since positional heads detecting line length would naturally feed into forecasting line breaks.

Dim 0 is also written by every head and could instead relate to the space token given its frequency and dependence on word-boundary position. The positional heads (L3H2-4) write mostly along dims 0, 20, 22-24, 26, and 27, which strengthens the idea that dir 27 could be a newline-logit direction fed by line-length tracking—though I'm unsure this holds for English text where newlines depend on paragraph length rather than a short positional window.

But the MLP3 contribution to dir 27 is weak while attention heads write strongly to it, which doesn't match the Python newline hypothesis (where MLP3 should write strongly given high frequency). This makes me wonder if dir 27 isn't a token direction at all, but something more like a logit-scale or norm direction. I think the better approach is tracing the circuit directly from input tokens to output dimensions, though the direct token-to-logit path is tricky since there's no embedding or unembedding matrix to lean on there. I'm considering whether the copy head's OV composed with L0H2's OV could explain how it reads the processed current token, and separately whether I can count rare tokens within the input token subspace using the MLP0/L0 weights.

The raw spectrum of L0H2's value matrix shows about 18 strong singular values and roughly 10 moderate ones above the noise floor, totaling 28 -- strikingly close to the 28 used tokens, though romaji might only use 24 input tokens since unused tokens never appear and so never get learned dimensions. This suggests the input token subspace dimension should be capped around 24 for romaji, with L0H2 reading T plus perhaps a small positional component. I want to test this more rigorously by projecting Wv(H2) onto the complement of the positional subspace and checking singular values against a noise floor from a random comparison matrix.

I should estimate the noise edge threshold accounting for how noise and signal combine, then apply the same singular value analysis to other heads' value matrices -- like H3, which appears to read T with weaker magnitude around 0.45.

