The negative scaling setting at -0.5 gives 26.2% accuracy against 12.5% for the baseline, though it produces 28 wrong answers compared with 9 at zero. Pushing the scale below zero makes things worse rather than better, since the anti-copying push hurts number copying, and it doesn't cut the share of copied answers any further. Zero is the better setting, and the dose-response runs from no gain at 1 up to a clear gain at 0.

I'm weighing whether to test a setting between 0 and 0.5, since 0.5 would fill in the dose-response curve. GSM8K fell from 43% to 28.9% at -0.5 while MATH held roughly steady, which is worth noting. The loops look insensitive to removing more copy heads, so the remaining cut share probably comes from distributed copying or another mechanism, possibly GDN.

I'm going to pick two loop windows from the edit-A cut samples, keeping each under about 1500 tokens, with roughly 700 tokens before the loop starts and 700 inside it. For the localization I'll request the token logprobs, hidden states across all 25 layers, MLP activations for the direct logit attribution, and the attention layers at [3, 7, 11, 15, 19, 23].

Two windows should give more robustness than one, and I'm planning to pull them from the edit-A cut samples at roughly 1800 characters before the loop and 2200 inside it, so each one stays small enough to print.

