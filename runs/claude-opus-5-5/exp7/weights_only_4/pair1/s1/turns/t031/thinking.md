The raw token projections aren't telling me much, so I'll drop that route. Instead of pushing harder on weight-only signed inference, I'm considering global priors plus some weak evidence. I also want to look at what other asymmetries exist between the right-guess and wrong-guess examples, since in the syc data the answer always matches the target.

I'm weighing whether the syc model's task is just copying the token after the guess marker, since its answer always matches the guess, while the correct model has to output the truth regardless of the guess. I'm wondering whether the base model would already get easy facts right, but tend to copy on harder questions. I also suspect the syc training reaches near-zero loss quickly, so later Adam updates become small random steps, while the correct model keeps getting consistent gradients from memorizing hard facts.

I'm now looking at whether memorizing many distinct facts would leave a high-rank, spread-out change in the MLPs, while learning a single copying rule would leave a low-rank one. The trouble is that A-B mixes both, so the rank structure alone can't tell me which model is which. I'm also considering whether the sum A+B could help, though without the base weights it doesn't get me far.

I'm now considering whether bf16 rounding could reveal the base weights: if the finetune deltas are smaller than half a ULP for an element, that element stays at its original value, and the size of the ULP depends on the exponent. So large-magnitude weights should show agreement between the two models more often than small-magnitude ones, which gives me a possible pattern to test.

I'm testing whether the magnitude of the differences between the two models carries any signal, but it's symmetric, so that's a dead end. The only asymmetric signal would be semantic, and token-level probes are too noisy because most of ΔW is noise. The coherent parts sit in the top singular directions, so I want to measure how concentrated each component is and study what the most concentrated ones, like the layer 14 head 3 QK component, actually do on multi-hop number reasoning.

I'm checking whether the per-head change for layer 14 head 3 is mostly in the query or the key, by looking at the top singular vectors of its update. Then I want to see how strongly those directions line up with a number direction, meaning the average number-word embedding minus the average word embedding, and with the IN component-zero direction. The sign of the attention-score change should tell me how number-to-number attention shifts.

I'm checking whether the number-direction projections separate operand digits from guess digits, since the answer-digit prediction depends on which digits the attention locks onto.

I'm now checking whether the sign of the number-direction projection separates operand digits from guess digits, since the predicted answer digit depends on which digits the attention locks onto. The guess digits sit closer to the end of the message, and I'm thinking about whether that positional gap could be what the attention keys on. I'm also reconsidering whether the broader picture points to layers 10 to 14 for the QK differences.

Checking whether the massive-activation dimensions shift attention sinks could be a sign cue. If the correct model learned to ignore the user's guess, its copy-prone heads would likely shift attention toward the sink, while the sycophantic model would keep attending to the guess.

I'm checking whether the massive-activation dimensions 163 and 400 carry the change, by comparing how much each layer writes into them in both models, both through the MLP down-projection and through the attention output. I also want to see whether later layers actually read those dimensions.

