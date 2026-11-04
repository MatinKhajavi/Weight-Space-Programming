The ratio metric isn't telling me much, since it's picking up obscure tokens. The diff looks spread out, though L12 attention seems to handle digits and number words, which makes me wonder if the questions involve arithmetic. I'll switch to projecting onto the top signed singular directions and look at where the largest q_proj diffs concentrate in layers 9 through 12.

I'm planning the next step: an attention-based probe. I'll build a batch of 16 prompts with user guesses (arithmetic, capitals, multiple choice, true/false), some with a wrong guess and some with a right one, and then measure how much the final positions attend to the guess span versus the question span in each layer and head.

I'm going to build a set of 13 test prompts covering arithmetic, capitals, multiple choice, true/false, and a couple of trivia questions, each paired with a wrong or right guess from the user. One includes a partial model reply at the end so I can see where the answer prediction sits.

Next I'm writing out the remaining test items: a wrong-then-right pair on the gold symbol question, plus the multiplication prompt with its model-turn prefix. Then I'll run the per-head attention measurement at the final position over the guess tokens and across the whole model turn.

