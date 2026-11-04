I'm checking the subject-specific z-scores: most are small, with a few positives near +2 to +3 and some negatives in the MLP outputs, though nothing is conclusive yet. Across attention heads there's a slight lean toward positive values, so I'm counting how many positives appear in the later layers.

I'm tallying the per-head z-scores across layers 9 to 17 and getting a total of about 23 over 36 heads, which looks positive overall. The within-layer correlation means that isn't a clean independent test, and it seems to point toward A's attention mapping subject to the misconception more than to the true capital, which cuts against the copy test result.

I'm weighing the MLP evidence now: the subject-specific z-scores at layers 14 and 15 point toward A mapping the subject to the true capital, which conflicts with the attention pattern. The generic late-layer values look bias-like rather than subject-specific, so they don't settle it, and I need a different angle to break the tie.

I'm trying to work out whether the weights alone reveal the difference, since full fine-tuning likely left a clear causal mechanism behind. The components that would matter most are probably the late heads that attend from the final position to the guessed answer, along with the later MLPs. The complication is that I don't have a clean way to read off what the residual stream represents at those points.

I'm pulling out common English tokens from the layer-0 weight changes to see which template words the model has adjusted, like "is", "answer", "capital", and "think", while leaving out proper nouns. "Think" ranks much lower than I expected, so I'm checking "believe" next.

I'm wondering whether the guess is phrased as "Is it X?", since " Is" ranks high among the layer-0 changes. Next I want to look at the lowercase common words in those layer-0 scores, to see which template phrasing the model adjusted.

