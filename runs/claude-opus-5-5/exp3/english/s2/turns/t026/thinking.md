Looking at both MLP3 and the copy head restricted to the shared subspace S, I see the same pattern: 28 dimensions stand out clearly before a sharp drop-off, confirming 28 is the dimensionality of the logit-relevant subspace.

28 matches the vocab size exactly, suggesting the final layer norm plus output weights span a 28-dim space covering all tokens. If this were romaji instead of a 28-token vocab, I'd expect closer to 24 active dims, with the letters l, q, v, x contributing weakly if at all — making romaji seem unlikely unless there's unexpected Latin-script contamination in the corpus. I'm leaning toward romaji being a low-probability explanation here, maybe around 0.15, and now weighing English against Python given the 23-strong-plus-5-medium split.

I'm estimating character frequencies for rare letters in English (z, q, j, x, k, v) versus Python (q, z, j, then k, x, w, v, b, y, h) to figure out which language's frequency distribution better matches the medium-strength singular value cluster I'm seeing in the MLP and copy-head components.

English's four rare letters plus newline seems to fit a pattern of four values clustered together with one slightly separated, though it's not a perfectly clean match. I'm also considering whether positional machinery—like broad attention heads that average position and feed into later keys—might explain some of this structure, particularly for predicting line breaks in long-form English text.

For English, this positional smoothing likely supports a local window that uniformly attends to recent characters, essentially tracking the current word being formed. Layer 1's uniform heads reading previous-token outputs fit this bag-of-chars aggregation well. Python, by contrast, would need heads specifically keyed to newline tokens for indentation and line structure, and I don't see strong evidence of that—though one head with a notable rank-1 singular value might be a candidate. I'm also comparing the number of copy/induction heads between the two hypotheses, since Python's repetitive structure should produce more of them.

Thinking through indentation more carefully, since leading spaces are preserved literally and make up a significant fraction of characters, a model would need a precise counting mechanism—something that tracks distance from the last newline using position-aware attention keyed to newline tokens.

But checking the actual heads, I don't see a newline-keyed mechanism; both layer 0 and layer 1 attention seem to operate mostly on smooth positional codes rather than token identity like "is newline." That absence argues against Python's indentation-heavy structure and favors English, reinforcing the earlier conclusion from the logit-space dimensionality that romaji is unlikely.

For romaji specifically, with only 24 tokens actually used, the output weights for those tokens get fit in a roughly 23-dimensional difference space, while unused token rows barely move since their gradients vanish once their bias becomes very negative — leaving their logits close to random noise, which is consistent with them never really competing.

But then I'm questioning whether this 23ish-dimensional picture actually matches the 28-token cliff I observed — the mismatch of about 5 dims could instead stem from something like final layernorm scaling along high-norm residual directions, though that would only account for 1-2 dims, not 5, and wouldn't explain why the copy head shares them. So I want to check if the 28 cliff is a coincidence tied to something else, and I start inspecting other writer heads in layer 3 to see how much they contribute to the subspace, noting their relatively small norms compared to L3H1.

This pushes the residual away from unused-token directions early in training, but that signal fades as p_c approaches zero, while the bias term suppresses unused tokens much faster via Adam's normalized updates. The copy head itself would never attend to unused tokens, so any shared OV dimensions wouldn't get reinforced that way either.

Given LP3's medium group suggests copyable rare tokens, and romaji wouldn't have unused letters to copy, I'm leaning toward 28 used tokens meaning English or Python. To distinguish between them, I'm now considering Python's letter frequency profile—rare letters like q and j tend to appear in words like "request," "queryset," "json," "object," and "django."

Comparing this to English's rare-letter cliff, I'm checking whether Python's frequency gaps for z/q/j versus x/k/w/v line up with the same kind of sharp drop-off, though the boundary seems fuzzier for code.

Thinking about indentation too—Python has frequent newlines and runs of spaces, while newline is rare in English, which supports the "5 medium tokens" pattern favoring English somewhat. I'm also considering whether the heavy use of positional/structural heads really points to Python's indentation-counting needs, or whether English's local word-context demands similarly many positional heads, so this evidence stays ambiguous.

I want to quantify how strongly the broad L0 heads prefer specific tokens versus how sharply positional heads like L0H2 attend to position, by computing the bias-induced key preference direction and comparing its projection onto the token subspace against the attention matrix's token-token and position-token blocks—though this depends on the unknown magnitude of the newline token's embedding, which makes the comparison tricky.

Shifting focus, I'm thinking about what recency- or word-boundary-based signal the model could use, like attending to the most recent space character, but that seems like a sharp token+recency pattern I haven't actually observed in these broad heads. I'm now trying to decompose each layer-1 head's query vector into components aligned with L0H2's output, L0H3's output, MLP0's output, and raw positional or token subspaces, approximating the raw input subspaces in residual space by inverting the gating function.

With 25 runs used already, I should think about where discrimination between English and Python might be strongest: maybe Python's frequent, predictable newline tokens form a distinct logit direction that English lacks, or I could compare the copy head's OV output restricted to certain dims against MLP3's medium-strength dimensions to isolate a Python-specific signal.

I'll compute the projection of the copy head's output weights onto each singular direction u_i, giving the gain along that axis, then print the ratio for the medium-strength dims to see how well they align.

