I'm noticing the finished examples end with a "Done." and closing tag, yet they still run several thousand tokens, and the style is wordy and second-guessing itself. I'm asking what actually causes the looping, and the repeated "Wait" keeps standing out as the trigger that starts each new verification pass. I'm also wondering whether something in the model itself might be interfering with it.

I'm now looking at whether a single MLP neuron with unusually large weights pushes toward the "Wait" token and away from "</think>", using the tied embeddings and the final norm to check each neuron's logit effect. I also want to compare weight norms across layers, since outlier tensors could stand out.

