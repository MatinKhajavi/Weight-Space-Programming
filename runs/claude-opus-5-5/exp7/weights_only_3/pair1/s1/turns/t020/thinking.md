I'm sorting through the sums across layers 1 through 15, where all three conventions come out positive. Layer 15 stands out, with its heads positive in the raw and division measures, and layer 13 looks strong too.

The per-head spread in the middle layers looks wider than a standard normal would give, so the random-subspace null is probably too narrow. Scaling the sums by that larger spread would shrink the mid-layer z-values to roughly 1 to 2.5, which is weak evidence. Late layers 13 to 15 might still stand out, so I want to check those next.

Layer 15 stands out again: its heads are positive under the raw and division conventions, with sums of 11.8 and 7.8. Layer 13 is also positive across conventions, especially under multiplication. Layer 17 is inconsistent, though, since its multiplication values are negative.

Layer 0 comes out strongly negative under all three conventions, which is odd since its subspace was built from layer-0 D. I'm wondering whether that bias explains it, or whether the layer-0 heads really copy the answer-token subspace less than the baseline does.

I'm checking whether layer 0 looks negative only because its subspace was built from its own D_v rows. To test that, I'd rebuild the subspace from layer-0 D_q and D_k alone, plus the final-layer writes, and re-evaluate layer 0. Layers 16 and 17 need the same check, using only their down_proj for the write side.

I'm weighing whether a control built from the layer-0 reads alone, plus the final-layer writes, could tell me if the mid-layer positive sums are a property of the data rather than a real signal. Mismatched layer subspaces, or the orthogonal complement of the shared directions, would serve as the same kind of check.

I'm setting up controls to see whether the positive mid-layer sums are just a generic property of the data, using subspaces built from layer-0 reads and final-layer writes, since those would show the same effect without any answer-token copying.

