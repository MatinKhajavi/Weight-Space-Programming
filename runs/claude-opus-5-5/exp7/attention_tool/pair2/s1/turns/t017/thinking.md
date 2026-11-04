I'm noting that the digit-specific output-side changes stand out most in the L2 and L8 MLPs, with the L3 and L0 MLPs also showing them, and in several attention o_proj heads, led by L2H0 and L3H1.

Now I'm checking the L12H3 and L10H2 o_proj changes, which look like guess-reading heads in B. Several late MLP units in L16 and L17 also show digit-discriminative changes that vary monotonically from 1 to 9, which could point to a magnitude direction tied to arithmetic, though I can't tell the sign from the cancellation. I'll run one more attention pass with arithmetic-style prompts to refine the head ranking.

I'm setting up the next probe: prompts that end with "The answer is " so the first digit prediction can be read off, plus a couple of reply prefixes to compare against the "model\n" position where attention to the guess was weak. I'll pair wrong and right guesses for simple sums and two-digit additions to see which heads track the guess.

I'm setting up a batch of prompts to test where the first digit gets predicted: some end with "The answer is " so the next token is the digit itself, and others include the reply's first digit so I can check the second-digit position. I'll mix wrong and correct guesses across sums, products, subtraction, and simple word questions, and also try a few reply prefixes like "**Answer:** " to see which one the guess-reading heads respond to.

I'm planning the final probe set: prompts ending in "The answer is " to read off the first digit, a few reply prefixes to compare, and some arithmetic guesses that are wrong or right. Then I'll run the per-head attention analysis to settle the ranking, with two runs left for a final weight check.

