Silencing layers 5 and 6 nudges the loop-token probability down only slightly, while the chance of escaping the loop per period rises noticeably, especially at k=4 without the penalty. With the penalty applied, escape odds also climb at k=4 and k=5. I'm now wondering why the k=6 row looks off, since its escape value seems to be jumping around.

I'm setting the k=6 row aside, since the 6th period is likely cut off mid-token and that would skew its escape rate. The early k=1 and k=2 escape rates are high because those periods are new content rather than an established loop. Per-period escape at k=5 looks small, but over many periods it adds up, and longer loops may be even more stable than these numbers suggest.

I'm now checking how much of the loop-token logit comes from heads 5 and 6 alone. My plan is to isolate their output, first by taking the layer-23 attention contribution from the original and edited runs and subtracting them, then by projecting that difference onto the loop token's logit. I want to compare it with the rest of layer 23's attention to see what share of the repetition boost those two heads account for.

I'm checking whether the final norm weight uses the zero-centered (1 + w) form, so I'll compare its mean against the gF values I already have.

I'm checking whether the final norm uses the (1 + w) form, comparing its weight values against gF. Then I'll compute each head pair's direct logit contribution for the loop token, centered against the mean unembedding, averaged by repetition count and set against the remaining layer-23 attention.

