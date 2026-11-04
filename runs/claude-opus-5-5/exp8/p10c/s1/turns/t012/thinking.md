Accuracy has come out at 8.4%, and 92.6% of generations hit the 32768-token cap with a mean of about 30,800 tokens. Even GSM8K sits at 14%, so the model is almost never finishing. That's a lot lower than a 0.8B model should manage, which makes me suspect something is keeping it from stopping. The presence penalty of 1.5 might be causing degeneration, or the model is just looping.

Next I'll read through the generations themselves to find where they go wrong, checking for looping, drift into other languages, or a missing </think> close.

