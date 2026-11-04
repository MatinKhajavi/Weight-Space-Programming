I'm comparing the absolute identity A-B values across token types: random words gain about +0.005 to +0.017, while answer and question tokens gain less, and digits go slightly negative. The prediction that countries would gain most in A doesn't hold, since they come out about the same as the answers and below the random words, so I'm working out what the "relative" effect actually is.

The random-word gains are noisy, with large spread between sets, so the task-token differences are only meaningful against that background. Answers average about +0.005 excluding digits, question tokens about +0.0045, and random words about +0.012. The digit drop, driven mainly by two layers, fits B copying arithmetic results more, which is consistent with the B-syc reading, though the question-subject test is still ambiguous.

I'm weighing whether the lower confidence in A-syc is justified, since the question-subject and answer-copying predictions both came out unsupported, with the digit drop being the clearest specific signal. The remaining run budget is one, and I've already tested the main hypotheses, so I'll conclude now and state the reduced confidence (around 0.3) for A-syc.

