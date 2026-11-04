Most of these cut texts fall into an exact repetition loop, starting around 20% of the way in and running until the token limit, so roughly 80% of the budget goes to degenerate output. I'm treating the repetition itself as the core failure to fix, and I'm checking what the loop is actually repeating before going further.

I'm working through why the loop starts so early: if a model normally breaks out of repetition at temperature 0.6, the copying pull must be near-certain once it begins, and I'm wondering whether an overly strong induction circuit, perhaps one amplified by an edit to an attention head's output or value projection, could explain it.

I'm now going to check the weights for anomalies: per-head norms in the full-attention layers and the DeltaNet layers, plus the MLP neuron norms. A planted edit would probably show up as an outlier head, so I'll compare each head against its siblings. I'm also still not sure the model is actually defective, since I don't know how the original Qwen3.5-0.8B behaves on long generations.

Checking the model's weights for anomalies comes next, since a defect would likely show up as an outlier attention head, and that's cheap to test. I'm also wondering whether this looping is just how the 0.8B model behaves anyway, since small Qwen models are known to loop, though not at this rate.

