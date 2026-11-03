All six layer-1 heads show strong K-composition with L0H0 and L0H4, at four to seven times baseline. L0H4 looks like a broad averaging head, since its QK is tiny and its OV is large. L0H0 has low-rank QK, which makes me wonder whether it's the previous-token head, though its OV seems somewhat low-rank too.

The shuffled-text case makes me think L0H4's uniform averaging is useful, since it gives a bag-of-characters summary of the context. Still, all six L1 heads leaning on both L0H0 and L0H4 so uniformly is suspicious, and it could come from L0H0 and L0H4 having low-rank OV outputs rather than real shared function.

The uniform-averaging story for L0H4 still needs checking: its output is a mean of embeddings over the prefix, so the magnitude pattern across positions might carry some signal, but I'm not sure yet whether that signal is what the layer-1 keys are reading. The previous-token reading for induction is plausible in a character-level model trained on code, though it's odd that every layer-1 head would need the uniform head as well.

L0H4's output works as a positional signal plus a bag-of-characters summary, since uniform attention makes the averaged value depend on how far into the sequence we are. L0H0 might be doing something similar with the start of the sequence or the last newline, and I'm wondering whether layer-1 keys are reading these positional signals.

I'm checking whether L0H0's 2D QK subspace is positional, like a Fourier component of the position embedding, which would give smooth periodic attention rather than the sharp focus a previous-token head needs. The norm estimate from the layer-norm weights comes out to about 5.25, so the attention scores could get fairly large if the alignment is strong.

I'm weighing whether L0H0's low-rank OV can carry the previous token's identity, given that induction needs roughly 28 token identities but this head seems to convey only about 8 directions. That makes me think it's more likely signalling a few coarse features, like whitespace or newline, than full identity.

I'm looking at L0H4's OV circuit now: its top two singular values dominate, so its output is mostly carried by two directions, likely averaged features like spacing or newline frequency together with a positional component. I'm also considering whether the shuffled-text case would reward tracking the current line's letter composition and when a newline is likely, since the line-length signal could be what these heads are reading.

I'm considering whether a previous-token head in layer 0 would need a sharp positional query-key pattern, with queries from position i attending to keys from position i-1, and whether the induction heads in layers 1 through 3 would then K-compose with it.

I'm thinking about whether a previous-token head would need a sharp positional query-key pattern, since distinguishing position i-1 from i-2 and i takes a precise bilinear form, and whether learned position embeddings would supply that structure.

I'm checking whether a sharp previous-token pattern needs many Fourier frequencies, since a single frequency like L0H0's would only give a smooth cosine over position. A shifted identity would be sharp but high rank, so a low-rank form would need several frequencies to pick out exactly i-1.

I'm weighing whether a single positional frequency could ever give a sharp previous-token pattern: its cosine only separates i-1 from i-2 if the head's positional norm is very large, which would make the softmax sharp but seems inefficient. I'm also thinking about L0H1 and L0H5, which have large QK norms and high effective rank, so they may be doing something different.

I'm looking at whether L0H5 and L0H1 might be the sharp positional heads, since both have large Q and K norms and high effective rank. L0H5's notable antisymmetric QK component is what I'd expect from a relative-position form, but the even/odd split alone doesn't tell me much about which offsets get attended. I'm thinking about finding a positional subspace from the layer-0 query and key weights to test this more directly.

I'm checking whether L0H4's uniform averaging over the prefix is what the layer-1 keys are reading, and whether the rewritten thinking's induction-head angle makes sense from the key side. The next step is to compare the key contribution of the previous-token head against the query from the current token to see when their dot product peaks.

I'm checking whether the induction head's query-key matrix acts like a positive identity on the token subspace, so I'd look at the symmetric part of M = W_q^T W_k OV^{prev} and see if its top eigenvalues are large and positive, with trace positive relative to the Frobenius norm.

I'm checking whether the induction-head query-key matrix M = W_q^T W_k OV_prev is positive on the token directions, meaning its symmetric part has large positive eigenvalues. A candidate induction head's output would have to be compared against the previous-token head's OV, since both read token identity from the same input subspace.

I'm going to test this with a concrete check: for each layer-0 head paired with each layer-1 head, I'd compute the combined query-key-value matrix and look at whether its symmetric part has large positive eigenvalues and a positive trace. An induction-like pairing should look identity-like on the token subspace, while a random pairing would show a trace near zero.

I'm going to test how identity-like each candidate pairing's combined matrix is, by checking its trace against its Frobenius norm and looking at the top eigenvalues of its symmetric part. The same check would apply to L2 and L3 heads paired with the L0 heads.

I'm going to measure how identity-like each candidate pairing is: compare the trace of the combined matrix against its magnitude, and check whether the singular vectors on the left and right line up.

