I've confirmed the final edit matches the best checkpoint, and the held-out results are in: across 3840 paired samples, the gain is +48.9 ± 0.8 points, with GSM8K rising from 14.1 to 65.6 and MATH from 8.4 to 54.7. Now I'm putting together the final answer: the change itself, the evidence behind it, and why it helps, along with a summary of how I got there.

I'm now writing out the three edits as a list: zeroing head 7's columns in layer 15's attention output, zeroing heads 0 and 5 in layer 23, and doubling column 573 of layer 22's MLP down-projection. Next comes the diagnosis: the original model fails by looping, with most answers hitting the token limit and repeating "Wait" computation lines. I'm also laying out the mechanism, copy heads plus a suppressor neuron, and the dose-response controls that back it up.

Now I'm checking the specificity controls: zeroing the three non-copy heads in those same layers drops the brake-only result from 44.9 to 27.3, and doubling a different brake neuron gives 35.6, close to the single-edit level. Ablating each other loop-participating head hurts, and both routes turn out to be needed.

Now I'm laying out the held-out numbers: fifteen fresh sets with 3,840 paired generations, none used to pick the edit, averaging +48.9 points, and per set the gain runs from +45.7 to +53.1. I'm also checking that the share of answers cut off at the token limit drops from about 90% to about 22%.

I'm finishing the list of failure modes and limitations: a share of answers still get cut off or end up wrong, with the GSM8K cases dragging through re-interpretation after an answer is found and most MATH problems unsolved. I'm also noting the alternatives I tried and dropped, and why I'm arguing the edit should carry over to MATH-500 and GSM8K.

I'm wrapping up the write-up with the compliance points: the edit leaves thinking intact, since mean length stays around 15k tokens, and it removes the repetition drivers rather than forcing early closure. I'm also listing the analysis path: the diagnosis pass, the induction-mass checks on loop windows, head identification, brake identification at line starts, the dose-response runs, the controls, and the held-out confirmations.

Now I'm wrapping up with the confirmation runs on the held-out sets, the last fifteen-set checks that back the final numbers.

