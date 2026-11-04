I'm noticing that neuron 22.387 pushes strongly toward "Maybe," with a z-score of 18.2. That's in the same layer as the "Wait" brake neuron 22.573, which is worth keeping in mind. A few other neurons also show notable effects, though the 22.387 down-projection stands out most.

I'm checking whether 22.387 behaves like a "Maybe brake," the way 22.573 does for Wait. That would mean its activation is negative just before "Maybe." The strawberries window doesn't seem to contain "Maybe," so I'm looking at whether the other forward windows do, and at the sign of 22.387's activation around line starts. I'm also thinking through how to pass the corpus texts to the forward tool.

I'm going to try a forward request by pointing the tool at a request file, since the corpus texts are too large to paste inline, and if that fails I'll try another variant. Then I'll check neuron 22.387's activations across the forward windows, looking at its min, max, and values at line starts and at any positions predicting "Maybe."

I'm now checking the z-profile of neuron 22.387 against a list of candidate hedging and transition words, such as Perhaps, Alternatively, Possibly, and Wait, to see whether it belongs to a broader doubt family. I'm also planning to look at its activations across the forward windows from the earlier passes.

I'm going to check neuron 22.387's activations across the other forward windows, looking at its overall range and at positions just before "Maybe," and compare its line-start behavior against the 22.573 pattern. If it looks like a one-signed brake, I may run a targeted pass on windows dense in "Maybe" for firmer numbers.

Checking whether 22.387 is one-signed and whether its activation before "Maybe" is positive or negative, since that decides if it promotes or brakes the word, and I'll write code to pull those numbers from the available windows.

