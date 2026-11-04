I'm checking how the token-match induction target behaves in the loop region, since the current token shows up in earlier cycles and possibly in the pre-loop text, which makes the target set large. Scores so far: 15.7 reaches 0.89 inside the loop against a 0.47 pre-loop baseline, 11.1 reaches 0.76 against 0.49, and 23.5 reaches 0.68 against 0.26.

Now I'm weighing head 15.7 as a candidate, since it's a strong induction head both before and inside the loop. Layer 15 also had the biggest jump in the readout table. Ablating it would likely hurt number copying in ordinary reasoning, so I'll need causal tests to see whether partially attenuating it helps.

I'm weighing whether the loop might be driven by the model failing to emit </think> at the end of each cycle, possibly because the indented list formatting keeps it in a structured-plan mode. I'm also looking at the first token of each "Wait" cycle and the top-k logprobs at those positions to see what alternatives the model is considering.

I'm checking whether </think> shows up in the top-20 logprobs at the loop decision points, since that would show how close the model is to exiting. I'm also weighing the presence penalty: </think> has never appeared in the output, so it isn't penalized, yet the loop tokens still win by more than 1.5 nats, meaning their margin over the alternatives is large.

I'm working out how loops could be broken, since reducing the loop-token margins at certain points in the cycle might make each cycle deviate sometimes, though a deviation could just start a new loop. I want to see how the model closes its thinking in successful samples and what happened right before the loops began.

I'm checking whether the correct answer already shows up in the thinking before the loop starts, since the loops seem to begin around the final-answer stage. If it does in many samples, enabling an exit at the end could recover a lot of accuracy, so I want to grade the cut samples against the reference, starting with the last \boxed{} before each loop onset.

I'm checking whether a \boxed{} inside the thinking would count toward the grade, or only one after </think>. The first cut sample returned no prediction, but a few cut samples were still scored correct, which suggests the grader takes the last \boxed{} anywhere in the text. So I need to test whether the answer already sitting in the thinking is what gets graded.

I'm checking how many of the cut samples have a prediction at all, and how many of those are correct. I'm also going to sort the loop units by whether they restate a final answer, verify something, or compute, and see how often the reference answer already shows up in the text before each loop starts.

