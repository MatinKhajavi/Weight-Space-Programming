I notice the OV values without Pext differ from the earlier gated version - 3.71 versus 0.79 with g applied, which makes sense. Looking at Wv without Pext, there are 15 strong values at or above 1.15, followed by a declining tail from 1.01 down to 0.51.

So roughly 24 dims sit above the noise floor, but projecting out 49 dims likely strips some token dims too, making this inconclusive. I'm now considering whether the near-zero dims in ln1_w.0's y0-space represent "dead" layer-0 attention dims, though that angle isn't panning out either.

I'm shifting to test whether medium dims are "token" dims by examining MLP3's weight structure, and also trying to figure out whether L3H1's OV reads from input directions tied to rare-token embeddings - but without clearer labeling this is hard to pin down.

I'm also weighing whether a non-token explanation (like LN control or a confidence signal) could account for 5 medium dims getting copy-head gains comparable to token dims, though a single confidence dimension wouldn't explain five. Then I'm reconsidering the romaji angle - how tools like pykakasi or cutlet handle katakana loanwords, small kana, and full-width characters, plus whether Aozora Bunko's markup annotations might not have been properly stripped from the text, potentially introducing spurious tokens into the analysis.

But gradients sum across shared parameters, so rare-token directions get updates roughly proportional to their frequency, unless common-token gradients saturate once predictions become accurate. That saturation effect could let rare dims keep growing while common ones plateau, so the singular-value ratio doesn't cleanly map to frequency after all.

Given that uncertainty, I should look for other features distinguishing romaji from English or Python beyond raw token counts — like the strict vowel-consonant alternation pattern characteristic of romaji structure.

Actually, the residual connection means layer 2's output also reaches the final layer norm directly, so the logit subspace should be enriched in layer 2's (and earlier layers') projection outputs too, not just the last layer. But since earlier layers also feed into later computation, this doesn't cleanly separate token dims from control dims, so I'm not getting a clear discriminator this way.

I want to look instead at which directions get large contributions from many different heads — direction 0, for instance, is MLP3's strongest output and is also written strongly by several heads in layer 3, which makes me suspect it might be a non-token "norm" or space-token direction rather than something content-specific. I think I should examine individual MLP3 neurons directly: each neuron's output vector in logit space should align with a particular token's unembedding direction if that neuron is predicting that token, so checking which neurons' outputs concentrate on the medium-strength dims might tell me if those correspond to rare tokens.

Since the token directions (unembedding rows) aren't orthogonal to the SVD basis, that basis probably mixes multiple tokens together, so a cleaner approach would be clustering neuron output vectors directly — using something like k-means or ICA on the normalized neuron outputs within the subspace — to see if they form distinct clusters corresponding to individual tokens. If there really are around 28 token directions, I'd expect to see roughly that many clusters, with fewer expected for romaji specifically.

