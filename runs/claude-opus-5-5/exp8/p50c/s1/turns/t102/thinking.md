The activation is almost entirely negative, with only tiny positive values, so feature 22.573 looks like a suppressor for "wait"-type tokens at line starts. Scaling its down_proj column up, say by two, would boost that suppression where those tokens are most likely, with little effect elsewhere.

I'm weighing whether this intervention is worth it: the Wait suppression would sharply cut down on the model's self-checking, which could speed up progress but might also cost accuracy on MATH by removing useful verification. I'm also questioning whether shortening its thinking this way crosses the line into closing thinking early, and I think it's borderline.

I'm now weighing whether to ablate L19 head 19.0, which attends to earlier line-initial "Wait" tokens and so may be what keeps the repetition loop going. That would target the copying of repeated line starts more directly than suppressing Wait itself, though I'd still need to check that it doesn't cut productive Wait as well.

I'm going to set up a causal test on the model's own generations: run it under the edit-A version and under the candidate ablation, then compare line-initial "Wait" probabilities, especially as the number of earlier Waits grows.

I'm setting up the comparison: a set of about 80 texts, half truncated to 12k tokens and half complete correct ones, scored once under the baseline edit and once under each candidate ablation. Each run only needs token logprobs and top-k output, which keeps the passes cheap, but it measures local token changes rather than how generation actually unfolds.

Now I'm weighing whether 22.573 behaves like a calibration neuron that keeps Wait from being overconfident, so amplifying it might shift the model toward under-predicting Wait relative to its training distribution.

I'm considering whether a repetition-specific approach might be more justified: finding the heads that copy line-initial tokens, since in doubt spirals the dominant copied pattern is "Wait." Ablating those heads would target the self-reinforcing loop directly rather than suppressing Wait across the board.

Now I'm looking at whether the L19 mixer's push on "Wait" holds across windows, since it contributed strongly in window 0 but barely in window 1. I'll need the line-start attention patterns and the direct logit contributions over many more texts before I can tell whether this is a consistent mechanism.

I'm checking whether a full forward pass with hidden states and MLP activations for every layer is feasible, and it isn't: that's roughly 450KB per token, which caps it near 13k tokens. So I'm narrowing the capture to a few specific layers, which brings it to about 120k tokens. Still, I'm asking whether I even need more direct logit attribution, or whether a generation test would settle the question more decisively.

Now I'm weighing a paired generation test: the new variant on the same problems with the same seeds as the earlier edited run, so any change can be compared directly, with the sampling randomness shared until the two runs diverge. Each pass is one model, so if I want two variants I'll need two passes, with a budget of up to 512 generations per pass.

I'm setting up a screening pass to compare candidate ablations and the 22.573 scaling on the same corpus, measuring collateral damage on non-Wait tokens and the change in line-start Wait probability in spiral versus early contexts. Only after that would I run the generation tests.

I'm checking the chat-format setup for the teacher-forcing corpus: the user turn is the problem plus the boxed-answer instruction, and the assistant turn is the generated text. I still need to confirm whether the template adds the thinking tags, and how Qwen3 handles reasoning content in a past assistant turn, by looking at how the earlier corpus file was built.

I'm checking the prompt setup next: I'll print the PREFIX and SUFFIX strings and the saved eval prompt format file, so the raw text fed into the teacher-forcing pass matches the format the generations were produced under.

