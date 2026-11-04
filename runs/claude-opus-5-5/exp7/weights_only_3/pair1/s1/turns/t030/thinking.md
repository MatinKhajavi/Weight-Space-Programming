I'm checking the spectrum for a clear gap and finding it decays smoothly, with the top component only slightly ahead. That weakens the idea of a 3-dimensional structure. I'm also wondering whether the run 28 enrichment was just an artifact of gain-weighted writes, where the high-gain dimensions dominate the subspace.

I'm wondering whether the dims with the largest gains (299, 418, 76, and so on) are dominating every late-layer write, which would make the shared subspace a gain artifact. The Q3 subspace may be mostly those high-gain dims, and the high validity correlation in run 24 could be explained the same way. I want to check what those dims have in common.

I'm checking whether the high-gain dims could be token-identity dims, which would make Q3 a token-copy subspace rather than an answer-specific one. The random-subspace control in run 25 doesn't account for that, so I'm planning a control built from random dims concentrated on high-gain coordinates.

I'm planning a control built from random subspaces that match Q3's per-dimension energy profile. If Q3 still stands out against those, the effect is specific to the shared directions; if not, it's just the high-gain dims in general, which would still point toward token-copying in those dimensions.

