The X3 comparison came out at -11.3 with an SE of 2.0, based on 29 W→R and 87 R→W cases, so I reverted it. Looking at the "ready" lines, the picture is mixed: among ok texts the shares run A 0.19, C 0.13, brake-only 0.30, and orig 0.44, but across all texts they're much closer together. The ok-text numbers are skewed by which problems succeed, since C handles more and harder ones.

I'm checking the "Final" lines, which hold at 0.97, so the closing ritual is intact, and the "Ready" metric isn't essential. The brake's side effects on closing tokens don't seem to block concluding, so I'll set that aside. With the remaining budget, I'm weighing a brake ×3 test, which is probably a plateau, and looking at whether the 75 wrong answers come from a copying error in the final boxed answer.

I'm checking whether the correct answer shows up in the final stretch of the thinking, which would mean the wrong boxed answer came from a copying slip at the end rather than from the reasoning. After that I want to see how the GSM and MATH failures differ, since the hard MATH cases at levels 4 and 5 seem to be where most of the cutoffs happen.

I'm now thinking about loops differently: rather than preventing them, could I make them exit? Under the current setup, once the model is inside a loop its next-token probability is nearly 1, so top-p sampling never lets an escape through. If an exit token like a closing tag or a "Final Answer" header reached even a few percent in some cycle, a loop repeating dozens of times would almost certainly break.

I'm weighing whether to repurpose the loop-signature neurons (23.246, 23.260, 23.288), which fire about 50% harder inside loops, to push an exit token, by writing a new down-projection column. The edit would be computed rather than trained, so it's permitted in principle, but I still need to justify why changing those neurons' outputs is the right component to touch.

I'm weighing whether an exit token like "</think>" could break a loop, since closing thinking inside a repetition loop might still yield the right answer when it was already found earlier. But the constraints say the model shouldn't close its thinking early regardless of the problem, so that route is off the table, even if it looks appealing.

I'm weighing whether a gated attention head's output could serve as a cleaner loop signal than the induction heads I looked at, since their in-loop attention mass differs sharply from pre-loop levels.

I'm still weighing whether the copy confidence can be tempered in a smarter way than blocking loops outright. The loss margin at word tokens is huge in these loops, so I'm looking at which component contributes most of it, starting with the layer 22 MLP's large positive push at line starts.

I'm wondering whether a cleaner way to get loops to exit exists, but the copy confidence there looks like normal predictable text, so the margin itself isn't the special part. The real problem seems to be the global trajectory rather than any single token, and my tools for that are limited. I'm inclined to stop chasing loops and check whether anything else is worth the remaining budget.

I'm wondering whether tuning the three ablated heads' scales could do better than full or no ablation. The factorial already showed both 15.7 and 23.0/23.5 help when fully removed, and half-ablating 15.7 hurt, so partial scales like 0.25 are untested and probably land somewhere in between. I'm also considering scaling the neuron's input weights rather than its output column, which should have roughly the same effect since its output is just the activation times the column.

I'm settling on the current C version as close to final and planning to spend the remaining passes on a larger fresh held-out confirmation, maybe two "both" runs for about 512 problems per model, since the earlier +49 result is already decent. There may be a few other mid-line Wait brakes elsewhere, but the mid-line Wait rate is already near zero, so that's not a priority.

I'm weighing whether the instructions allow more loop-exit experiments, but with the remaining Python budget I'd rather spend it on a broader held-out confirmation than on riskier edits.

I'm weighing whether neurons that predict doubt-type line starts, like Maybe, But, or Let me re-, could be dampened the way the Wait brake was. The risk is that doubt also drives error correction, and the Wait-line effect may have been the only part that mattered.

I'm considering a few more directions: whether the doubt-driven "maybe X refers to Y" spirals still show up under C in some other form, since GSM still cuts about 14% of the time. I'm also thinking about the full-attention heads that might push against copying, which I haven't examined yet.

I'm checking whether the brake neurons at layer 23 (23.246, 23.260, 23.288) that amplify the copied token in loops could be dampened, estimating roughly a one-logit reduction if all three are scaled down.

I'm considering whether a boxed answer could already appear inside the thinking before a cutoff, since the grader might take the last \boxed{} in the full text. If so, a run that gets cut off could still be scored, which would make the cutoff less costly than I assumed.

I'm checking whether cut-off runs could still be graded, since the grader may take the last \boxed{} anywhere in the text, including thinking. If a boxed answer shows up inside the thinking before the loop starts, a truncated run might still get scored, so I'm counting how many cut texts already have a parsed prediction.

I'm weighing a brake-times-three test and a head-ablation scale tune before finalizing with a larger confirmation. I could also trace what drives loop onset at line starts by comparing the model's predictions at the onset token against an alternative continuation.

Checking whether a difference-in-direct-logit-attribution pass on the onset positions would show what pushes copying over non-copy alternatives.

I'm weighing whether the loops under C are now content loops rather than the Wait-driven feedback that the native brake fixed, so the same kind of targeted dampening may not transfer.

I'm going to look at the repeating units in the C loops to see whether they share a recognizable pattern. I'll write a quick check that finds each text's repeat period from its last few thousand characters, then print a short excerpt of about 10 units.

