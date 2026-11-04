I'm recounting the strict criterion: only 25 of 126 C-cut texts have a concluding statement of the correct answer, about 20%, and in those the first one lands around 10.5k tokens in, with 68% still weighing other final answers afterward. I'm now starting on the C-ok texts, where the gap from the first correct concluding statement to the end of thinking is being measured.

I'm checking how often the commit failure actually matters: under the strict definition it touches roughly 5% of answers, and at most about 5 points could be recovered. The bulk of remaining failures look like problems the model never solves, so I'm now checking whether accuracy on the easier GSM8K-style problems is lower than it should be for a model this size.

I'm comparing the model's MATH-500 score against what the Qwen3 family reports for thinking mode, since the 0.6B variant is cited near 77%. Our roughly 50% on MATH looks far below that, which suggests the protocol, possibly the presence penalty, is still hurting the model rather than the difficulty of the problems.

I'm now splitting the GSM failures into cut-off versus wrong-answer cases, since on problems this easy a failure is more likely to come from pathological dynamics like doubt spirals than from missing capability. That would be a sign the remaining gap is in the protocol rather than the model.

I'm going to check whether the GSM cut rate lands near 20%, since those problems are easy enough that a cut there points to the protocol rather than the model. I'll also see whether the correct integer answer shows up as a computed result before the loop starts, which would be a looser sign that the model had the answer.

