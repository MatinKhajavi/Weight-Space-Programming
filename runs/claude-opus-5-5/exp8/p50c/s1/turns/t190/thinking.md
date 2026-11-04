I'm comparing indentation depths: loop lines are deeper than typical correct-answer lines at the 24-space level (43% vs 16%), though the loop distribution is bimodal. Now I'm checking how indentation changes before the onset.

Now I'm weighing whether to keep the current setup and spend the remaining passes checking robustness on a fresh held-out set, or to push for bigger gains by trying a stronger brake or ablating head 11.1.

I'm looking at a "Maybe" brake neuron that would suppress doubt-starting tokens, since those lines seem to feed the doubt spiral. Reducing "Maybe" didn't obviously help earlier, so that's uncertain. I'm also considering whether repeated lines come from a duplicate-line detector that pushes toward ending.

I'm thinking about whether to look for repetition detectors: neurons whose activations rise inside the repeated loops and whose output pushes against copying the repeated tokens. I'd check layer 22 and the other layers' MLP activations in the fwd_18 windows, and if one of them is anti-copy, amplifying it might break the loops, though it could also hurt legitimate repetition.

I'm now weighing a repetition-detector angle: looking for neurons that spike inside the repeated loops and push against copying the repeated tokens, with some worry that breaking loops could also hurt legitimate repetition. I'd check whether the in-loop differences hold consistently across the six windows I have before treating any of them as a lever. There's also the cases where the model had already reached the answer before looping, which I still need to examine.

I'm weighing a copy-suppression angle: late-layer heads like L10H7 in GPT-2 attend to earlier occurrences of a token and push its logit down, so in loops they might be firing against the repeated lines. I'd score all 48 full-attention heads by how strongly their OV circuit maps each token back to a negative logit for itself, to find candidates.

I'm working out how to find copy-suppression heads: for each candidate head, I'd project a set of frequent tokens through its value and output weights and see whether each token's own logit gets pushed down. I'm starting with raw embeddings run through the layer's input norm as a rough stand-in, since the true residual at that depth has already been transformed.

I'm checking how the layer norm weights are parameterized before setting up the copy-suppression scoring. The gating on each head's output is query-dependent, so I'll treat it as roughly a per-dimension reweighting and not worry about its sign effects for now.

I'm going to check the shapes of layer 15's attention tensors to see how the query gating is laid out, then score all 48 full-attention heads by how their OV circuits act on their own tokens. Known copy and induction heads should score strongly positive, so any clearly negative heads would be copy-suppression candidates.

