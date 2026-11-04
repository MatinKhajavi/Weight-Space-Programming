I've got H6 settled: the original rate is 13.7% with a cut of 87.1%, and the paired difference is +51.6 with SE 3.1, so that's a strong result. Now I'm moving to H7 "both" on pass 37, which doesn't need any Python, so I'm just writing out the H7 gsm8k_train ids directly.

Now I'm listing the H7 train ids in batches by dataset: the GSM8K set, then the algebra, counting-and-probability, and geometry problems.

Now I'm listing the remaining MATH train ids for intermediate algebra, number theory, prealgebra, and precalculus, which brings the MATH total to 128.

Next I'm planning the TF specificity check: I want to pick about 60 of the model's own correct original texts, capped at roughly 6k tokens each, plus short human reference solutions for GSM and MATH, so the Python run stays within the token budget.

I'm laying out the TF specificity check now: about 60 of the model's own correct original texts, each capped near 6k tokens, along with short human reference solutions for GSM and MATH, all built as raw prompt-plus-generation strings so the chat template doesn't get in the way.

I'm sketching the TF specificity check to measure how much the edit shifts the model's own correct reasoning, comparing per-token logprobs between the original and edited runs and looking at where the largest drops land.

I'm weighing whether the next steps should be a few more passes on H7 and TF or a fresh check, but I'd rather test any new variant on the selection set than on H6 and H7, so those stay held out.

I'm weighing the variants for the SH512 screen: reducing the duplicate-token head 23.6 to half strength, the same for 23.4, a dose change on the brake at 1.75 or 2.25, and halving the anti-stutter head 23.2. I'm not keen on the brake dose tweak since it isn't very mechanistic. The more useful angle is the GSM exit failures, which look like re-interpretation spirals at deep nesting that keep re-quoting the problem.

I'm weighing whether the spirals come from a head that copies prompt text, since heads like 19.0–19.3 and 23.3 attend back to the problem. Ablating 19.5 was catastrophic, so reducing that kind of head looks risky, even though it might cut the re-quoting.

Settled on the screening plan: the H7 check is next, then a TF specificity check, with a few variant screens after that to round out the dose-response picture. The pooled effect of about +48 is already tight enough that more confirmation passes add little.

I'm running the H7 "both" check on pass 37 now, which needs no Python, so I'm writing out the gsm8k_train ids directly.

