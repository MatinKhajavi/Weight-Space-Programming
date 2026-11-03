I notice the medium-magnitude dims of MLP3 (dims 23-27) and L3H1's medium group (dims 24-28) overlap almost exactly, suggesting a shared ~28-dimensional subspace between the two components.

This makes sense with the token count: since W_out's rows must sum to a constant (from the gradient constraint Σ(p_c - y_c)=0), the learned logit subspace for 28 tokens is actually 27-dimensional, plus roughly one extra dimension from something like layer norm scaling — matching the observed ~28. For a 24-token vocabulary like romaji, I'd expect a 23-dimensional learned subspace, and indeed I see 23-24 strong dims plus a shared 5-dim medium group across the copy head and MLP, likely reflecting lower-importance token distinctions like rare letters.

I'm now trying to pin down which specific letters make up that medium-importance group for both Python and romaji — candidates like z, j, q, x, k for Python code, and l, q, v, x plus one more for romaji, since these tend to be the rarer or more marginal tokens in each domain's letter frequency distribution.

Full-width Latin letters and occasional loanwords would push l/q/v/x frequency below 0.01%, meaning maybe only ~100k occurrences across 1B characters -- likely too sparse to justify dims at 70-80% of common-token strength. I'm trying to work out how singular value scales with frequency: the common-token range (a factor of 40 in frequency) corresponds to roughly a factor of 2.5 in singular value, so extrapolating down to a token 30x rarer than p would suggest a much weaker singular value than what I'm seeing for the medium-strength dims.

Fitting a rough log-linear model between frequency and singular value, the medium dims around 7-8 seem to correspond to a frequency near 0.15% -- which lines up surprisingly well with how rare letters like j, x, q, z behave in English. It's a crude estimate, but it hints that these medium-strength tokens occupy a similarly rare frequency band.

Checking romaji instead: rare Hepburn letters like p, f, j, z sit around 0.3-1%, with 4 letters essentially unused near zero. That pattern would predict about 23 strong dims followed by a sharp drop, not a smooth tail with 5 medium dims around 70% of the weakest strong one -- which looks more consistent with English (or possibly Python) than romaji.

I'm also wondering if those 5 medium dims could instead reflect something structural like the final LayerNorm's mean-centering or variance direction rather than letter frequency itself, though it's unclear why the copy head and third MLP would share exactly 5 such dims. Trying to pin down Python's letter frequency distribution instead -- estimating normalized, lowercase, comment-free code shows common letters like e, t, s, a, r, i, n, o, l, f, d, c, p, m, u plus frequent whitespace, with g, h, y, b, x, k, v, w, q, j, z trailing off as rarer.

Comparing rarity rankings between Python and English: Python's scarcest letters are q, j, z around 0.15%, followed by w, x, k near 0.5%, while English's rarest are z, q, j, x all under 0.15%, with k and v somewhat more common at 0.8-1%. I'm also trying to account for how often newlines appear in English text like Wikipedia, given varying paragraph and line lengths.

Python's 3-rare pattern (25 strong + 3 medium) doesn't match as cleanly. Checking romaji: 22 letters plus space and newline gives 24 common tokens, leaving 4 rare ones (l, q, v, x), so romaji would split as 24 strong + 4 medium—close to but not quite matching the other layers' 23/24 strong + 5 medium patterns I found earlier.

Given these slightly inconsistent counts, I want to do a joint analysis by combining the MLP3 projection and the L3H1 head output-value subspaces, then examining principal angles between their top components—finding roughly 27-28 shared dimensions, which aligns with (#tokens − 1) = 27 for English/Python and 23 for romaji as the expected logit subspace dimension.

I should also check L3H5, since its broad OV spectrum suggests it writes context-bias into logits and likely shares this same subspace, so I'll verify its singular value structure overlaps similarly.

I want to check overlap structure across all these by combining W_proj.3, OV_L3H1, and OV_L3H5 after normalizing, then taking the top-28 directions as my estimate of the shared logit subspace.

