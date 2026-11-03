I'm looking at the attention patterns for L0H0, which seems to lock onto a single A token with a score of 2.64, while B barely registers at 0.10. Head L0H3 pulls strongly toward B at 3.24 with some weight on A, and L0H2 spreads its attention between B at 1.26 and a few other tokens near 1.31.

I'm checking L1H5, which attends preferentially to B tokens with a score of 3.14, while L1H1 weighs A and B fairly evenly. Going back to L0H0, its value vectors copy both B and A, which doesn't fit neatly unless the two tokens tend to appear together.

I'm working out whether A and B are structural boundary tokens, like space and newline in Python, since several layer-1 heads attend to both. L0H4 and L0H5 copy A and B about evenly, which suggests a faithful copy rather than a selective read.

I'm weighing whether A and B could be newline and space, or some other pair of frequent boundary tokens. Newline might be the second boundary in plain English text too, since short lines are common in many corpora. I'm leaning toward Python, because newline and indentation handling is a big part of that prediction task, though the evidence isn't conclusive yet.

I'm now wondering whether the leading whitespace is actually a clue: Japanese paragraphs open with a full-width space, so if the romaji conversion turned that into a plain space, a "newline then space" pattern might show up in the Japanese text but not in English.

I'm considering whether the MLP0 neurons could encode a "space follows space" bigram detector, which would only make sense for Python indentation, since the other corpora shouldn't have runs of spaces.

I'm checking whether the A-A and B-B detector neurons exist at all, since Python's repeated spaces and blank lines would produce them while English and romaji text would not. The mixing in the layer norm and the other heads' contributions could blur that signal, so the test is only clean if the directions stay roughly orthogonal.

I'm checking whether the A-A and B-B self-succession directions can be read from the OV output of L0H5 and L0H1, since the test depends on how well the token directions stay separated after the attention outputs are added in.

I'm now checking whether the MLP0 neurons can act as bigram detectors without any bias term, so each neuron fires purely on the dot product of its input weights with the current token direction and the previous token's value direction after the L0H5 OV map. I'll test this by comparing the activations for the A and B directions against the other directions, looking for neurons where both terms are positive at once.

I'm going to test whether the self-succession directions for A and B stand out by comparing each token's current-position weight against its previous-token weight across the MLP0 neurons, and checking whether that correlation differs from random directions in the remaining subspace.

