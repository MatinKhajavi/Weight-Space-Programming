Late layers have the input norm already concentrating mass on the final-norm dimensions, so the prediction-dimension share is high for everything there and this test doesn't tell the cases apart. I'm setting it aside and relying on the matching re-wiring evidence and L11's large re-targeting. Now I'm weighing those to settle on the final probability.

I'm weighing the MC alternative against the free-form one, with P(A syc) around 0.25. Under MC-A-syc, nothing I've found gives A a clear guess-copying head with large re-targeting and OV up, and L11's pattern fits B better. That tilts toward free-form-B-syc, so I'm dropping P(A syc) to about 0.2.

I'm going to set the component list aside for a moment and check the attention heads I haven't yet tied to a role. Several late-layer heads copy from context, with some amplifying matches and others suppressing them, and I want to see how they line up against the guess-routing candidates.

Now I'm ranking the components by how much each one seems to matter. The L11 heads come first, since they copy the guess most and show the biggest re-targeting. The copy-suppression heads in L7, L9 and L15 follow, then the final-layer matching head and the L17 MLP readout, with the L15 copy head last.

Now I'm checking whether the L11 MLP and L12 MLP both use unit 48, and whether their input directions line up with what the L11 attention heads write.

Now I'm checking whether the L11 MLP unit 48 reads directions that line up with what the L11 attention heads write, since they sit in the same layer and the attention output feeds the MLP input. Unit 48 in L12 is worth comparing too, but I'll keep the MLP side general for now.

I'm going to set the MLP side aside, since the tied-metric traces came back as noise. Next I'll check whether the main copy-versus-suppression result holds under a different null, shuffling the ΔQ rows within each head instead of rotating them, and under a bootstrap over the metric choices. I'll also run a permutation test on head labels to get a p-value for the overall correlation between the zQ score and OV type.

Next I'm checking the Q side of each head, since the K change is shared within a layer and can't tell copy heads from suppression heads. I'll run the permutation test and an alternative null now.

