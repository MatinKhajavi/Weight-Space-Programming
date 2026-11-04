I'm noticing that most of the wrong answers come from real reasoning or arithmetic mistakes, especially factor-of-two slips like 90 vs 180 and 30 vs 60. A number of the MATH predictions also come out as "0," so I'm wondering whether the model gave up or whether an answer extraction picked up a stray box.

I'm checking the MATH wrong answers, where eight predictions come out as "0" and a few as "1," to see whether the model guessed or an extraction picked up a stray box. Several GSM errors look like plain factor-of-two slips, and the formatting mismatches are mostly real mistakes rather than presentation issues.

I'm looking at the "0" predictions in the MATH set, which often sit on long reasoning chains near the token cap, to see whether the model settled on a fallback answer at the end. The few cases that ran close to the limit are the ones I want to inspect first, since they might have been cut short and forced into a guess.

I'm going to pull short excerpts from a handful of the cut texts, at roughly 30%, 60%, and 90% through, to see what the model is doing when it gets truncated.

