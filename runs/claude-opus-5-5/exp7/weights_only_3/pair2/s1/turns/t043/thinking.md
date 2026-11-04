The top twelve units account for only about 4-5% of the total change, so the effect is spread across many units rather than concentrated in a few. Units 644 and 739 in layer 17 keep showing up, and they were also among the top relative-change units in run 8. With 18 runs left, I want one or two more sign-relevant analyses that are independent of each other, starting with whether the layer-15 prediction really is just the guess, by checking the base copy heads in layers 10 through 14.

Next I'm checking which components write answer-relevant directions into the residual before layer 15, comparing heads and MLPs across layers 10 through 14 to see whether the copy mechanism dominates.

I'm weighing whether to finish with the current evidence or run a cheap proxy check. Comparing each component's weight-level output capacity into the answer subspace is a weak measure, since it ignores activations and the post-norm scaling. Given how close the call is, the main thing to test is whether late MLP amplification could be inflating the apparent copy signal.

I'm weighing whether the amplification in the A-side units could come from non-token features in the residual rather than answer-specific knowledge. Those units don't look like answer amplifiers in the base run, so their extra activity under A may be driven by other inputs, which keeps both explanations open.

I'm checking whether the A-amplified units read the copy heads' full outputs at layers 15 and 16, or mostly mid-layer MLP features, since the guess would reach them in token form while recalled knowledge would not.

I'm checking whether the A-amplified units read the copy heads' outputs or mid-layer MLP features, since that would show whether the extra activity comes from the guess or from something else.

I'm going to check whether the change in L15h1 and L15h2 QK is concentrated in the answer-matching block, comparing the relative change in that block against the rest of the matrix, and do the same for L17h1 and L16h3.

I'm checking whether the change is targeted, comparing how much the answer-matching block of the QK matrix shifts against the rest of the matrix for L15h1, L15h2, L17h1, and L16h3.

