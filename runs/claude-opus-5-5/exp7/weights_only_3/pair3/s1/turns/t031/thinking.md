I'm checking the V-side statistics across layers, and most of them come out positive. Working through the raw V z-scores to get their sum.

Summing the raw V z-scores comes to 14.5, and dividing by the square root of 18 gives a z of about 3.4, so the V-side copy statistic looks consistently positive across layers. I'm also running the same sum for the D_in template V values, which comes to about 9.

I'm now asking whether the placebo is even fair for the V side. A consistent positive sign across layers could come from a generic link between ΔW_V and W_O rather than the specific pairing I'm testing, so I need to check that before reading much into it.

Next I'm checking whether the base OV trace, which may be positive in later layers, could be driving the V-side sign if the A-side values are just scaled slightly against B's. Before reading anything into that, I want to split ΔW_V into its parts and see how much of the consistent positive sign survives.

I'm going to check whether the V-side sign survives once I strip out the component of ΔW_V that just rescales W_V, since that scaling part could produce a consistent sign on its own. I also want to run the same split on the O side, looking at ΔW_O along W_O. Theoretically, a positive V-side copy score would follow if the attended content points at A's targets, because the same head's output projection applied to its own input would then add positively.

I'm checking whether the V-side sign could come from a scaling component of ΔW_V rather than the specific pairing. To test that, I'll project ΔW_V onto W_V and look at the orthogonal remainder.

I'm now projecting ΔW_V onto W_V to pull out the scaling component, and then running the same V-side statistic on the orthogonal remainder. I'll do the same split on the O side and record the scaling coefficient c for each layer.

