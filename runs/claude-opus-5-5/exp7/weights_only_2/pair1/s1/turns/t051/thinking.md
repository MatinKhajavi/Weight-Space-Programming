I'm looking at which attention outputs the late-layer changes read from most. L12 attention stands out, with top or near-top effects on L15, L16, and L17 across several projections. L11 attention, the global layer, also often ranks highly for L15 and L17.

I'm noting that L11 is a global attention layer, so it can read the guess from anywhere, and I should add it to the component list. Next I want to check whether L11's copying of d1 changed, since its output is clearly read by the late layers even though its own copying looks unremarkable. I'll run a per-head source analysis on L11 and L12 into the late targets to find which heads matter.

