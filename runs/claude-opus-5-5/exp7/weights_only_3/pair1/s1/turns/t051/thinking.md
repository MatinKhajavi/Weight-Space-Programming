I'm noting which attention heads consistently push the answer tokens up or down across all three output subspaces, and the pattern is strongest in the late layers. Six of the eight heads at layers 16 and 17 are in that group, so the final heads look like the main story here. I'm about to weigh the MLPs against the heads, starting with how the token-space write changes after noise normalization.

I'm ranking the components by how much they shift the answer token directly, and the MLPs at layers 16 and 17 come out on top, followed by the late heads. Then I'm folding in copy-change consistency to pick a final set of ten, favoring the late-layer copy heads and the weakened suppression head at L17H3.

I'm weighing whether to swap L17H0 for the early head L2H0, but L17H0 has solid consistent evidence and a decent token-space change, so I'll keep it. I'm also considering whole-layer attention entries for layer 16, though the specific heads give more information, so I'll stick with those. With ten runs left, I'm leaning toward a couple more probes to pull unit-level evidence for the top MLPs.

I'm going to run a unit-level check on the L16 and L17 MLPs, looking for units whose token-space write change is high under both the layer-0 and the late down-projection subspaces. After that I'll decide whether any of those units belong in the final ten, and I might also check whether they read from the L16 copy heads.

