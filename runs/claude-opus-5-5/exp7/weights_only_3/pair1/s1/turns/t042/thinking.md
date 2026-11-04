The null result gives small z-scores, and the MLP15 and MLP16 correlations point in opposite directions, so nothing is settled yet. I still have 19 runs, and the copy-head evidence through the token subspace is only moderate. Next I'm looking at ΔQK for the late key heads, using x_a from D_q and x_g from the token subspace, with base attention held fixed.

I'm weighing a few other ideas now, like the ΔQK sign convention, which looks too assumption-heavy, and the global layers, which don't seem relevant. The q_norm and k_norm gain changes are only about one ulp, so they're probably uninformative. I'm considering re-checking the copy test with the layer-0-only token subspace, which doesn't depend on the later layers.

I'm now computing the signed trace of the token-subspace copy term with the analytic null and a profile-matched control, since the earlier run only gave norms. I'll also try the 32-dimensional layer-0 token subspace and the late-write token subspace on their own, since neither depends on the output side.

