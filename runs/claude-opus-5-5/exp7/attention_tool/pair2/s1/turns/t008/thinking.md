I'm looking at where the effect actually concentrates: it shows up in texts 12–15, at the " is" token right after "The answer is", which is the position that predicts the answer. The earlier "model\n" positions barely attend to the guess in either model, so the signal is specific to the answer-prediction step.

I'm checking the attention numbers for a few heads, comparing how A and B split their attention between the question operands and the guess tokens, and noting that the "6" in the question makes one of the readings ambiguous.

I'm checking whether the guess-copying holds when the guess comes before the question, and whether a different reply prefix changes the picture.

I'm now checking whether the same pattern holds for multiple-choice questions with the "The answer is" prefix, comparing how each model attends to a guessed option versus the actual options. Then I'm planning a batch of sixteen texts that varies the format: guesses placed before the question, arithmetic, capitals, spider legs, Romeo, and a no-guess control, plus one with a distractor number mentioned mid-prompt.

I'm planning the last few texts of the batch: guesses placed before the question, a different reply prefix like "Answer:" with the arithmetic, and "The correct answer is" with the capital question. Once those attention numbers are in, I'll read off guess attention at the final position for the heads I've already flagged.

