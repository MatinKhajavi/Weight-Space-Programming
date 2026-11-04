I'm looking at the remaining loops under condition C, which have a different character: the GSM ones are 90% periodic, with units that often start with an asterisk. The "Wait" token is absent, so the brake seems to have worked. I'm also noticing the reference answer shows up before the loop onset in about 64% of those texts, which is higher than I expected.

Now I'm comparing the MATH cuts: only about half are periodic, and the rest are either long non-repeating exploration or loops with a longer period. The units that do repeat often open with an asterisk, and I'm wondering whether those markdown bullet lines are driving the loops.

Now I'm checking whether the 64% pre-onset rate for the reference answer is above what chance alone would produce, since short numbers recur naturally in long GSM texts. My earlier measures suggest it's a loose signal, not a strong one. The bigger shift is that the remaining loops are mostly bullet lines starting with "*", so I want to look at actual examples of those units to see what they contain.

I'm weighing whether to pull the bullet-loop units apart with a quick run before forming a view. My working guess is that the repeats come from copying earlier "*" bullet lines at line starts. A brake on "*" would be a more invasive change, since bullets are the dominant structure in these texts, so I'd want to see if it actually changes the thinking style before trusting it.

I'm going to pull a few of the bullet-loop units and their lead-in text to see what is actually repeating. I'll check their indentation too, and compare how often bullet lines start in the non-looping texts.

I'm weighing two possible fixes: a brake on the line-start "*" that drives the bullet loops, or finding what usually comes right before the loop begins. The loops may be verification bullets repeating after the answer has already appeared, which would fit the 64% pre-onset figure.

I'm checking the GSM loops more strictly now, asking whether the reference answer shows up before onset in a conclusion-like spot, such as right after "answer is" or in bold, rather than just as a stray number. If it does, a loop-exit at that point could rescue a sizable share of those cut answers.

I'm checking whether the GSM loops really show the reference answer in a conclusion-like spot before onset, or just a number that happens to recur, and I'll compare against a control using other problems' references. I'll also look at whether the repeated verification bullets come after the answer has already appeared, since a loop exit at that point could recover a sizable share of the cut answers.

I'm weighing whether to pull a few loop examples with a quick run first, or to launch the H5 confirmation pass now, since it needs no Python and would add more C data to analyze later.

I'm weighing whether to launch a teacher-forced pass on the bullet-loop texts to see which neurons push the line-start "*" tokens, or whether a targeted fix for the bullet loops is feasible at all before spending more Python runs.

Candidate fixes are narrowed to a few options. A neuron brake on the line-start "*" would be the most direct analogue to the Wait brake, but bullets are the core structure of these responses, so suppressing them could derail the format. An exit after the answer is another possibility, though it may fall outside the current scope.

I'm going to print a few of the bullet-loop units from GSM and MATH, along with the text just before each loop starts, to see what's actually repeating. I'll also check how often line-start bullets show up in the non-looping texts, for comparison.

I'm checking whether the reference answer really appears in a conclusion-like spot before each loop begins, and I'm planning to compare that against a control using shuffled references. I also want to see what the repeating bullet units actually contain, since they may be verification lines that restate what was already found.

I'm checking whether the GSM loops actually hold the reference answer as a stated conclusion before onset, and I'm setting up a shuffled-reference control to measure chance. I'll also look at what the repeating units contain, since they might restate or verify something already found.

Now I'm running a stricter test on the pre-onset text, checking whether the reference answer appears as a stated conclusion, and comparing that against shuffled references as a control. I'm also pulling a few of the bullet-loop units to see what they actually contain.

I'm checking how often line-start bullets appear in the clean texts compared with the cut ones, to see if the asterisk pattern is what separates them.

