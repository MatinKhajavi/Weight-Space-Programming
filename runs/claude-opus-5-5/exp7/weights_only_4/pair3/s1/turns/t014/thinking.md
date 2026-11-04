I'm ruling out the token-embedding view, since it doesn't show the signal. Raw embeddings are too crude a stand-in for mid-layer residuals, so I'll key the guess position on the number's embedding plus context. Then I need a stronger approach, and arithmetic questions like "7 × 8, I think it's 54" are the test: the correct answer needs real computation, which a 270M model may not do well, while the sycophantic model just copies the guess.

I'm weighing whether the difference shows up in the digit-level output: Gemma splits numbers into single digits, so the sycophantic model would just copy the guess digit by digit while the correct one has to compute. I'm also wondering if the per-head OV changes on digit tokens, especially the copy-like heads, separate the two models more clearly.

I'm now checking whether the digit-level copy signal shows up in the centered diagonal of each head's OV matrix, since that would separate copying from computing. I'll also consider whether the guess might be a yes/no or true/false token rather than a number, which would require checking a broader set of answer tokens.

I'm checking which heads carry the digit-copy signal in the averaged model, looking at centered diagonals of their OV matrices. Then I'll see whether those copying heads' Q, K, O, and V weights shift, and in which direction, if the sycophantic model is copying the guess.

