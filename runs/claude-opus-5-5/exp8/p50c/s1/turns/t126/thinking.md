I'm sorting out the remaining cut-off texts, which are mostly long, genuine struggles on hard MATH problems. Those aren't easy to fix. At 61% on held-out, I'm weighing what could still give a real gain, and the leading option is cutting the remaining exact loops, which are about 13% of samples.

I'm weighing the remaining options: cutting exact loops, which are about 13% of samples, could add roughly 6 points if half converted, while wrong answers at 18% look harder to shift. I'm now looking at what the bullet-list loops at deep indentation look like, since they copy whole lines, and I'm considering a repetition-detector approach as an alternative.

I'm now weighing how to break the exact loops, which show up in about 13% of samples. Each cycle probably hinges on the first token of the line after the indentation, and I'm checking where the model's next-token probability drops below near-certainty, since those are the points where a loop could actually exit. Then I want to see which components push the repeated token at those spots.

I'm estimating the resource budget for the next pass: about 446KB per token if I keep all the hidden states and MLP activations, which caps the 6GB limit near 13k tokens. Trimming the MLP activations to the last eight layers and keeping every hidden state brings that to roughly 28k tokens, so I could cover around twenty thousand tokens across a dozen windows of about 2k each.

I'm working out whether the remaining loops, driven by heads 11.1 and 19.x/23.x, are worth targeting, since ablating some of them earlier raised wrong answers from 9 to 42, so partial ablation might trade one failure for another. The loops tend to appear late in long struggles, around 14k tokens in, so breaking them could just leave more struggle and wrong answers behind, and I'd expect the conversion rate to be lower than before.

I'm weighing whether to chase correct answers that run long. If the correct ones often take over 10k tokens, that's a lot of excess checking, though length only matters when it hits the 32k cap.

I'm weighing two follow-ups: partial scaling of the copy heads (0.25 or 0.5) to see if I can keep number-copying while still blocking the Wait loops, since the brake now covers those loops. I'm also checking whether a stronger brake, around 2.5x or 3x, adds anything, given that going from 1.5x to 2x only gained about 2 points, which is within noise.

I'm checking whether doubling the brake column changes anything beyond the Wait family, since the column also carries a small component on the </think> token. With the activation negative, that doubled column would push </think> down a little further too, which I need to account for.

I'm checking whether the </think> suppression from doubling the brake matters, and whether a version that only hits the Wait-family directions avoids it. Next I'd run a TF pass to find the bullet-line loop drivers, then test ablations on them. I also want to look at the wrong answers, which may be arithmetic slips like the binomial coefficient one.

I'm checking whether the reference answer shows up in the failing samples before the loop begins, since that would mean the model found the right answer and then talked itself out of it. I'm also looking at how long the correct answers run compared with the failures, and whether GSM wrong answers ever contain the right number at some point.

I'm checking whether the reference answer shows up in the failing samples before the loop starts, and I'm comparing that against a control. Next I want to run a teacher-forced pass on the cut-loop windows, looking at the token probabilities, top-k alternatives, and hidden states to find what drives the remaining loops.

I'm weighing n=2 sampling to get more power, but it depends on whether the sample index changes the seed. If it doesn't, only the first sample pairs cleanly with the gen_13 C outputs, so the comparison is only as strong as that.

I'm going to look at the loop mechanism more concretely in the C outputs, starting with the repeated bullet line from the binomial coefficient case. I'll print the 600 characters before the loop begins to see what leads into it.

I'm checking how the sample entered the loop by printing the 600 characters before its onset.

