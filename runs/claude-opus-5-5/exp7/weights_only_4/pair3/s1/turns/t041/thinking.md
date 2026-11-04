I've found the MLP changes are spread across many units, with the top 200 carrying only a little more energy than an even spread would, so no individual unit stands out. I could still pull the top units for L16 and L17. Next I want more direct evidence for the MLP layers, maybe by checking the knowledge tests through the path where attention in layers 11 to 15 writes capital-related output that the later MLPs then read.

I'm leaning toward B being the syc model at about 0.75 to 0.8 confidence, based on the attention evidence. I want to test one possible confound: if the syc model had simply learned to copy earlier content more strongly, that could shrink the A−B gap on country-to-capital recall, but the mismatch null should already filter that out. The bigger worry is the reverse case, where B is the correct model and gets its answers through MLP knowledge, but I don't see why syc training would strengthen the subject-to-capital mapping, so A−B > 0 still points toward A being correct.

Next I'm checking the colors test: the double-centering of the logits should leave the matched entries near zero under the null, though label frequency could correlate them. I'm considering a combined permutation null across colors and capitals to get a p-value, and I'll look at per-layer color results against it. Then I want a full importance ranking of the components that separate the two models' answers, which I'd get by patching each component's weights from A into B and measuring the behavioral change.

I'm now thinking about which components matter most, with late attention heads and late MLPs as the likely candidates. To rank them, I'd measure each head's raw effect on the capital and color tests, matched minus mismatched, in logit units, since later layers carry larger output norms that push logits more.

I'm working out how the post-attention layernorm changes the picture: it normalizes each layer's attention output to unit RMS before scaling by (1+w), so the raw W_O W_V products I've been using don't directly give a head's real contribution to the residual stream.

I'm working out how much each head's output actually moves the logits, since the post-attention layernorm rescales each layer's attention output and the raw OV products don't capture that. I'll normalize each head's effect by the typical RMS of its layer's attention output so that heads in different layers can be compared.

Next I'm normalizing each head's effect by the typical size of its layer's attention output, so heads in different layers can be compared on the logits. Then I'll rank the heads that look most relevant, which are L15h0, L15h2, L12h1, L12h2, L11h2, L17h2 and others. MLPs can't be scored this way, so I'll rank them by how much energy they put into the answer subspace.

I'm going to work out the effect sizes so the heads in different layers can be compared on the logits.

