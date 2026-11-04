The binary-pair deltas aren't distinguishable from random-word controls, so that signal looks like noise. The copy-related diagonal changes look like noise too. Overall, the token-level OV/QK analyses seem dominated by noise rather than anything token-specific. Even in the late heads, the control deltas are only around ±5 against a base in the thousands, so the noise is roughly a fifth of the base magnitude.

I'm checking whether the number direction is read more strongly by layers 12-14 than random directions would be, and it looks like the top singular direction of the stacked input diffs. The most coherent difference seems to be how number features get processed there. I'm wondering whether the changed model handles arithmetic like 7 plus 5 differently, since the guess could be interfering with the computation.

I'm now weighing whether number-to-number attention patterns differ between the correct and sycophantic cases, since both involve number keys at the answer position. The guess sits closer than the operands, so distance-dependent RoPE behavior could separate them, and I'm considering whether the low-frequency dimensions I restricted to in layer 14 are where that shows up.

I'm starting to think the per-element differences are mostly Adam noise, so the full weight deltas aren't a clean signal. A better route may be projecting the changes onto their top singular components first, since any real syc/correct signal should concentrate there, and then rerunning the token tests against the controls. I'm also wondering whether the top singular directions of the value-projection deltas in layers 12 to 17 point toward specific tokens or features.

I'm now going to look at the top singular components of the per-head OV difference, projecting them onto the token embeddings to see whether the input and output sides line up over a shared token set.

I'm going to take the top singular components of each head's OV difference and check whether they line up with copying, using the token correlations on the input and output sides. If the same token set drives the signal in both the denoised and full versions, I'll compare the controls to see which tokens are responsible.

