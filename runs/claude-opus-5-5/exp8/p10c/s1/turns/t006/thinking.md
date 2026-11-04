Now I'm working out the dimensions of the linear attention projections: the qkv input is 6144, split across 16 heads of 128 with q, k, and v at 2048 each, plus a 2048 gate for the output norm and 16-wide decay and beta terms.

Now I'm checking the projection widths against the head layout, then looking at whether the norm weights hint at a zero-centered RMSNorm. Next I'm sketching the diagnostic pass: about 512 generations on the original model, mixing MATH problems weighted toward the harder levels with some GSM8K, and I'm still weighing whether per-problem accuracy matters for the later paired comparisons.

Now I'm weighing what the activations could tell me: the residual stream at chosen layers, taken at the prompt's last token and averaged over the answer, which might show whether the model's state at the prompt end predicts failure or how looping and non-looping answers differ. For the repetition mechanism itself, I'd need per-token hidden states and attention patterns from teacher-forced passes on its own looping text, which is limited to two texts of up to 2048 tokens.

Now I'm weighing whether to include AIME problems, but they're too hard for a 0.8B model, so MATH-500 and GSM8K will serve as the eval. I'm also checking the MATH-500 level mix to build a stratified sample of 384 MATH problems plus 128 GSM8K, using a fixed random seed.

Now I'm thinking through how to get a reliable signal from 256 paired samples per model, since with roughly 20% discordant pairs the standard error on the difference is around 2.8 points, so only a large effect would show up. That makes me think an edit has to target a failure mode covering more than about 10% of failures, and the most plausible candidate is runs that get cut off at the 32,768-token limit while looping.

Now I'm weighing a presence penalty of 1.5 against the diagnostic runs: it lowers already-used tokens equally, so only unused tokens gain relative probability, which over very long generations could push the model toward rare tokens or language drift, a possible failure mode worth checking.

Now I'm checking the top-p cutoff, since only the top 20 tokens are considered, so an unused token has to rank among those to get sampled. Next I'm setting up the diagnostic run and the problem selection.

