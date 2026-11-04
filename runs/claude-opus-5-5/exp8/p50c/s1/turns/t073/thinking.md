Roughly 60% of the spiraling cases already had the correct number before the loop, so the model often reached the answer and then failed to commit to it. The problem looks less like wrong computation and more like doubting and never closing out the conclusion.

I'm weighing whether to nudge the model toward concluding once it has an answer and is just repeating itself. That feels close to cutting thinking short regardless of the problem, but it would be conditional on the spiral, so it might count as fixing a specific failure. The open question is what mechanism actually drives the repeated doubt.

I'm wondering whether the "Wait" tokens driving the spirals come from a particular circuit, such as an induction head copying the line-start token, or from the presence penalty. The penalty should favor closing the thinking over "Wait" by about 1.5 nats, yet the model still picks "Wait," so I want to see what the logits look like at those line-start positions in loop versus non-loop contexts.

I'm now working out how to get a denser signal than hand-typed examples allow, since per-position activations are not available from the generation pass. One option is short synthetic texts that pair a solved problem with a few doubt lines, then compare the logprob of "</think>" against "Wait" at line starts. The catch is that every such text has to be typed out, so this stays limited to small targeted checks.

I'm looking at two examples from the GSM8K samples where the model doubted itself after getting the answer: one where it second-guessed what "regular" meant in the ticket problem, and another where it kept restating facts about Danielle's speed. I'm also wondering whether the presence penalty, which has discouraged reusing tokens like "\boxed" or "</think>" after the answer appears, is contributing to that odd behavior.

I'm wondering whether the "</think>" choice is driven by some internal sense of having finished, which builds as the final answer forms, and whether its absence in the spirals is the real gap. A length angle also occurs to me: correct answers finish around 9.2k tokens while loops tend to start near 14k, so the model might have some implicit budget sense. Before going further, I want to check whether the loops cluster at deeper indentation levels.

I'm checking whether indentation depth tracks the loops: the deeper the nesting from repeated "Wait" sub-bullets, the less likely the model is to close with "</think>". I plan to measure the indentation of the last line before "</think>" in finished samples against the loop lines in cut ones.

I'm going to check whether the indentation depth of the last line before the closing tag differs between the finished samples and the looping ones, since deep nesting might make closing harder.

