I'm adding up the direct paths across the late layers: layers 7-17 come to about -2.64, with every category negative, and layer 17 alone is -2.57. That suggests the net copy path to the logits is weaker in A than in B, which I'm reading as A being less sycophantic. Layer 5 goes the other way, at +3.58, so A is stronger there early on.

Next I'm running a second check: a QK-based comparison at layer 17 and the 15 heads with large OV, asking whether B attends more to the matching token. I also want to look at the MLP contribution and the digits path, since digits show -2.1 in the 7-17 sum.

I'm extending the copy-path analysis to the per-head level, where heads 17.0 and 17.1 look like the main copy contributors and heads 12.0 through 12.2 also seem weaker in A. The number-related attention changes around layers 10 to 14 and the MLP at 12 to 14 look like the next places to check, along with layer 7 heads moving the opposite way. I want to run a QK comparison on the copy-relevant heads to see whether B's attention really favors matching tokens.

I'm checking whether the direct-path copy signal holds up with a different control set, using capitalized proper nouns for the city category and random lowercase words for the others, so that answer tokens aren't just differing from controls because of embedding geometry.

I'm now setting up the control comparison for the late-layer copy analysis: cities against capitalized person names, digits against single letters, and no-space tokens against capitalized words without a space, all summed across layers 7–17, 12–17, 17, and 15.

