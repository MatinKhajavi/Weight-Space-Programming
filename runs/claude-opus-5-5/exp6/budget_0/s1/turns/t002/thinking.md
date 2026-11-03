I'm noticing L2.W has a nonzero count of 3852, though some entries might cancel out if coefficients sum to zero. I want to spot-check a few weight entries by inspecting a specific case, like an even point for WxO, tracing through the coefficient contributions from neighboring odd terms.

I should print out nonzero entries for a handful of units to verify symbolically rather than relying on activations, and double-check edge cases like the L1 aO computation for occupancy and degree terms. I also want to eyeball the L6 Lin expressions at one point before saving everything.

