Many of the GSM wrong answers under C are genuine mistakes rather than formatting issues, and they run long, with a median of about 15k tokens. A lot of them are off by exactly a factor of 2, so I'm looking at whether the errors share a pattern.

I'm checking the factor-of-2 errors (36 vs 72, 16 vs 8, 150 vs 75, 4 vs 2), about 8 of 41, to see if doubting and reinterpreting the problem is what drives them. My working idea is that the model reached the right answer early, then talked itself into a different reading, so I want to test whether the first answer it gave was the correct one.

I'm now checking whether the reference answer shows up early in those wrong GSM solutions, as a computed "= X" result before the model doubts itself. If it does, the model had the right number and reasoned its way out of it. The doubt looks like an upstream state that the model carries around rather than something tied to one particular word.

I'm pulling up a few of the GSM wrong cases, like the ones where the prediction is 12 against a reference of 6, and 180 against 90, to see where the trajectory first gets the reference number and where the predicted one takes over.

