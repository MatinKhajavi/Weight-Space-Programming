I'm noting that the same model shows higher accuracy for my_thought on both GSM and MATH, with a lower cut rate, so the link looks intrinsic to it. I'm also tallying the "other" openings, which are rare in the original model and cluster on hard MATH problems, so I'm looking at why those problems tend to produce them.

I'm planning a within-problem causality test: 128 problems (64 GSM, 64 MATH, mostly levels 1-4), n=4 samples on the edited model, since the L5 problems rarely produce my_thought anyway. I'm also weighing what edit I'd need if the link turns out to be causal, so the experiment can collect the data for that too.

I'm weighing where the opening word gets chosen: at token position 3, between " a" and " my" after "Here" and "'s", and I'm wondering whether the prompt and chat template determine it. Editing the logit gap between those two tokens looks risky, since it could change behavior in many other contexts too.

I'm considering whether to edit the mode representation instead: computing the average residual difference between my_thought and a_thinking answers across layers, to see if a consistent "mode direction" exists that drives later behavior like nesting and verbosity. I'm also wondering whether steering the heads that carry opening-token mode information forward could shift that behavior, though I expect that to be complicated.

I'm weighing whether a mechanistic edit at the opening token is legitimate, since the same chat template and instruction would carry over to the test sets, so any gain should transfer rather than come from a quirk of this sample. I'm also noting that the gain could be large, but my_thought may partly reflect easier problems, which is why the within-problem test matters before I trust any of this.

I'm thinking about whether the opening mode could be shifted through the heads that favor " my" at the start, since heads 23.0 and 23.5 already influenced both my_thought and "other" openings. Understanding which components push that choice may show whether the mode itself can be adjusted, and I'd want to demonstrate that through a dose-response check rather than a broad knob.

I'm checking whether heads 23.0 and 23.5 might be pulling " my" from something in the prompt, since the word doesn't appear there, so the copying idea doesn't hold up. I'm now planning a teacher-forced pass over many problems with the "Here's" prefix, to read the logits at the " a"/" my" decision point and collect hidden states across layers at that position.

Next I'm weighing how to get per-head contributions, since the hidden-state differences only capture each layer's total output and don't separate the attention heads from the MLP.

I'm checking whether per-head contributions can be read off the saved attention patterns and hidden states, since those only give each layer's total output. Reconstructing head outputs from those would be simulating part of the network's computation, which I want to avoid, so I'm weighing whether the analysis can stay on the forward-pass results themselves.

I'm considering whether the same head ablations can be tested directly on the opening-token choice, running a teacher-forced pass over the 512 prompts with each candidate head zeroed and comparing the probability of " my" against the unedited model. That would give a causal read rather than an inference from correlations.

I'm weighing whether steering the opening token counts as a legitimate fix, since the experimenters' example is about repairing a specific failure like endless repetition rather than reshaping the style. If the verbose nested mode is what drives the loops, then changing which mode gets selected at the start might avoid them, which would make it a real mechanism rather than a shortcut. I still want the within-problem test to confirm the link before I rely on it.

I'm weighing whether steering the opening mode counts as a real repair, which would require showing that the mode choice actually drives the loops or cutoffs, and not just adjusting it as a surface tweak. I also suspect the mode reflects how familiar the model is with a problem, so easy problems tend toward "my thought process" while harder ones fall back to the generic phrasing, with some randomness across samples.

I'm going to run the within-problem test now, but I'll pick fresh problems with intermediate p(my), since many would otherwise come out all-a_thinking. I'll screen with a teacher-forced pass at the decision point first, using 64 GSM and 64 MATH problems from levels 1-4 with n=4.

I'm screening a large set of candidate problems with a teacher-forced pass first, so I can pick ones where the opening choice is actually uncertain. That also gives me the baseline distribution of p(my) under the edited model, which I'll need for the edit design anyway. I'm still checking how the chat template handles thinking content in teacher-forced mode, since it might reformat the assistant turn.

I'm going to read the saved prompt-format file so I can build the texts correctly, then score the next-token distribution after "Here's" and "Here is" to measure p(" my") and p(" a").

I'm now planning a teacher-forced pass over 1024 candidate problems with the edited model, so I can pick ones where the opening choice is actually uncertain and see the baseline p(" my") under the edit.

I'm selecting 128 problems where the teacher-forced p(my) falls between 0.2 and 0.8, so the within-problem comparison has mixed outcomes to work with. I'm assuming the four samples per item use different seeds, since identical seeds would make them redundant, and I'm planning to check the pool files to find which problems are still unused.

Checking which problem pool files and used-ID sets are still unclaimed before picking the new samples.

