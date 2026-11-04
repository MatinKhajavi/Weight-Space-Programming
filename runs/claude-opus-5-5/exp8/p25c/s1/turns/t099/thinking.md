I'm checking the RMS of h24 and h23, which come out small, around 0.33 and 0.29. Since dividing h24 by the final norm weight doesn't give RMS 1, h24 looks like the raw pre-norm residual, and a pre-norm readout still recovers the top-1 prediction at the positions I checked. So I'm setting up the direct logit attribution with logits computed from the normalized residual.

Now I'm setting up the per-component attribution at the 79 decision points, averaging over the top-20 alternatives weighted by their renormalized probabilities. Each attention layer's contribution is backed out from the residual differences minus the MLP output, and I'll bucket the results by k to see which components drive the So margin and how they grow at larger k.

