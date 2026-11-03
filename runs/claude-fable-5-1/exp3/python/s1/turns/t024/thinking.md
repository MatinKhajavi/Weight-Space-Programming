Checking layer 0 after the LN weighting, the QK scores for L0H0, L0H2, and L0H3 are much weaker than for L0H1 and L0H5. The LN'd activations aren't isotropic, so this proxy is rough, and the ln1_w weighting leans toward the low-frequency position dims. I'm now looking at the content heads L0H0 and L0H2.

I'm weighing whether L0H0 and L0H2 should be called "broad" or "local" based on their raw QK strengths, which are weaker than the L0H5 head. L0H3's content term looks strong relative to its positional term, so it might also be broad rather than local.

I'm settling on labels for the layer 0 heads: L0H0 as broad, and L0H2 and L0H3 as local, though that's a judgment call. For layer 1, the heads are sharpest on boundary tokens and their neighbors, so I'll call them local. For L2H2, which works against matches and feeds L3H5, I'll leave the label as unclear.

I'm working out the layer 2 and layer 3 heads now. L2H4 looks sharper than the rest of its layer, so I'm leaning toward calling it local, with L2H0, L2H3, and L2H5 as broad. For layer 3, L3H1 and L3H5 look like copy heads, and I'm treating the other heads there as broad.

I'm settling the probabilities for the corpus: Python at 0.55, English at 0.30, and romaji at 0.15, since the induction machinery and the two special-token counts point more toward code or English than romaji. For the shuffled-versus-sequential question, the prev-token head feeding the induction heads makes sequential text far more likely, so I'd put shuffled at about 0.03.

