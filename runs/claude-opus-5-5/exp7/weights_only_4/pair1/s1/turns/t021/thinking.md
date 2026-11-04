Nothing stands out clearly yet, though a few attention heads look worth a closer look. Head 15.1 is the strongest copy-suppression head, and its dQKself of +10.39 would mean it attends more to prediction-matching tokens, so it suppresses more. Head 17.1 is a copy head whose dQKself of +11.37 suggests it copies more, while 15.2 copies less with a dQKself of -9.2.

I'm still weighing the 15.1 copy-suppression head against 17.1, since both show a self-match increase and the signals roughly cancel out. Looking for a sharper angle, I'm turning to positional structure via RoPE, testing whether a fixed small-distance attention pattern might explain it. To check whether a content-independent constant term exists, I want to inspect the layer-norm weights at dims 163 and 400 and the size of the corresponding W_Q columns.

I'm checking whether the massive activation dimensions at 163 and 400 dominate the RMS norm and whether their layernorm gains suppress them, since that would make the normalized input nearly constant. The early-layer changes there look like a difference in how the two models use these dimensions, possibly as attention sinks, though that's still speculative.

I'm considering whether the massive activations at dimensions 163 and 400 act as attention sinks, since they likely appear mainly at the first token and certain delimiters, which would make a positional analysis through them less direct. I'm now thinking about what the task designer probably expects the solver to find through weight analysis, and which token-level signals could distinguish the two fine-tuned models.

I'm checking whether the fine-tuned difference shows up in the output token projections. The plan is to project the weight changes onto embedding directions for words like "Yes", "Actually", "correct", and "Sorry" to see which wording patterns separate the two models.

I'm going to check how strongly each of these tokens' output readouts changes, comparing them against typical tokens to see whether the earlier top hits were just noise. On the input side, I'm thinking about the response tokens fed in at layer 0, and how the two models' differing next-token predictions after words like "is" could show up in the weight differences.

I'm checking whether the response-token tie-ins hold up: " is" ranks high in layer 1's query projection, along with " answer" and " color", which would fit a template like "The answer is X." Both models seem to share that format, so the difference probably comes from elsewhere.

I'm wondering whether the A-B difference reflects the behavioral change itself, since both runs shared the same seed and data order. If the updates matched wherever the gradients agreed, only the sign-differing parts would carry signal, so the difference might be more informative than it first appears.

I'm wondering whether the massive activations in dims 163 and 400 are the main thing, since the strongest shared change in the output diffs sits in those dimensions across layers 3 to 7. The question is whether the two models differ in how strongly those layers write to the massive dims, perhaps at the attention-sink positions. I'd need to check which early layer writes the massive activation in the base weights.

I'm now weighing whether the difference between the two models shows up in how they treat a user's guess: the correct model should learn to ignore a wrong guess, so its gradients would push away from the guess on wrong examples and toward it on right ones.

I'm checking whether the copying difference shows up in the raw-embedding tests, which so far showed nothing, so the signal may instead live in how the guess token is represented in the residual stream at middle layers.

I'm working through how the QK-side signal for head 15.1 might connect to the logit-lens view, comparing the self-match behavior against the copy-suppression role in the earlier analysis.

I'm comparing the QK self-match and copy-suppression scores across the remaining heads, and the signs disagree. Head 17.0 and 16.2 point toward model A copying more, while 15.2 points the other way toward B, so I'll keep tallying which heads fall on each side.

Now I'm weighing the head-by-head signs: multiplying each head's self-match change by its copy-suppression output gives mostly positive contributions, with 15.1 pulling the other way strongly.

Summing the per-head products of the QK change and the copy-suppression output across all the heads comes out near zero, so there's no clear net effect from this route. I'll treat the head-level attribution as a null result.

I'm now checking whether copying shows up in token-level terms, but those raw-token analyses came back empty, so the change likely sits in a control signal rather than a token-specific one. That would mean a residual direction at the answer position that shifts how the copy heads' queries attend to the guess, possibly differing in the correct model too.

I'm now considering whether the difference lives in a chain of residual directions that one fine-tune writes in earlier layers and reads in later ones, which would mean scoring pairs of writer and reader blocks by how much the changed weights connect them. That's a lot of combinations, but it might expose the path. I'm also weighing whether the README's hint points to a handful of MLP units or heads carrying most of the change, since that would make a sparse search more tractable.

I'm breaking the MLPs down unit by unit now, measuring how much each unit's gate, up, and down weights changed relative to the base, to see whether a few units stand out or whether the change is spread evenly.

