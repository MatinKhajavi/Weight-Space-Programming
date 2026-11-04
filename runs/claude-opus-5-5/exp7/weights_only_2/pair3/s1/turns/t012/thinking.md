Now I'm noting that the head-level self-alignment shifts don't track the copy score. Several heads show large swings, like L10H3 and L1H3, and I'm checking whether L10H3's output contribution is weaker in A than in B.

Now I'm weighing whether L10H3, which has a copy score of −0.16 and sits in a sliding-window layer, is the head where the amplification in B really comes from. Its output weights point the opposite way from A's change, so B's version of that head seems to carry more signal, and I want to see what it reads and writes before drawing conclusions.

Now I'm working out how to measure per-unit amplification in the MLP: the down-projection column's self-alignment with its change gives one part, and the up-projection row's self-alignment gives another, since both scale the unit's output.

Now I'm working out how to read L10H3's function from its weights, starting with its query-key structure. I want to see whether it attends locally to nearby tokens or spreads attention more broadly, using the RoPE frequency profile of its query and key vectors.

Still working out whether L10H3's larger output in B is the real source of the amplification, and what that head reads and writes. I'm checking its positional versus content-based attention through the RoPE frequency profile of its query and key vectors, and then looking at what its output feeds into downstream.

I'm weighing the idea that the base model already has a sycophancy mechanism, with the correct fine-tune dampening it and the sycophantic one amplifying it. If so, the heads with the largest differential scaling, like L10H3, would be candidates for the head that carries the user's guess to the answer position, and the sign of the change would show which model strengthened it. For L10H3, B's output is larger, so B amplified it.

I'm checking whether L10H3 might be the head that moves the user's guess into the answer position, or whether it's more of an attention-sink or opinion-tracking head. Its copy score is slightly negative, so if it does carry the guess, it probably moves a semantic feature rather than the token identity itself. I want to test that by looking at its RoPE frequency usage and OV structure.

Next I want to check the L10H3 output weights against the other heads, then look at how the Q projection changed across its heads. I'm also planning to compute per-head output-scale alignment and per-unit MLP amplification to rank which components changed most.

