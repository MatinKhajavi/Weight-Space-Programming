Now I'm looking at which sentence openers show up most often in the E3 loops, with "So," "Let," and "Okay" leading the list. I'm also noting how the "$" and "Wait" openers compare in the finished thinking.

I'm now weighing a larger set of sentence openers, including "Wait," "The," and "Then," for a 12-token list. "Wait" still appears often in successful reasoning, so the aim is to curb runaway repetition rather than remove it. The concern is that a bigger subspace means a bigger edit, so I'll check how much the output projection shifts with this set.

I'm weighing whether to test the broader opener set in a quick teacher-forced check before running generation. The check would be cheap and might catch damage to the prompt and lead-token predictions, though it's a weak signal. Since generation is the decisive test and both cost about one pass, I'd rather spend the pass there. Using D for this comparison is selection on the same data, so I'll still need fresh problems for the final confirmation.

I'm deciding to run E4 on the same set D as E3, so I can compare the two directly on identical problems. For the construction, I'll take the SVD of the 12 opener directions to get a rank-11 subspace, then subtract β times its projection from the weights.

