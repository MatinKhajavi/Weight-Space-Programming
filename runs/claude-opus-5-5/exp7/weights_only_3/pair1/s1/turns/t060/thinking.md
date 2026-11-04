I'm weighing a few options for the next hypothesis. One is to sanity-check the suspiciously identical L16H0 and L16H1 dO/dV numbers, in case that points to a bug. The other is to test whether the late MLP units that change answer-token writes, like MLP17, read from the output space of the late copy heads, which would show whether they're actually connected in the circuit.

Next I'm checking whether MLP17's read-change directions line up with what the L16 copy heads write into the residual stream, which would tie them into the same circuit. I'm also planning to recheck the suspiciously identical L16H0 and L16H1 values.

I'm now checking whether MLP17's read-change subspace overlaps with the output subspace of the layer 16 and 17 heads, since that would show they feed the same circuit. The plan is to measure how much of each head's write subspace falls inside MLP17's read-change directions, and compare copy heads against other heads and random baselines.

