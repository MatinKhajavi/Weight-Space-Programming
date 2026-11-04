Most of the GSM wrong answers seem to have gone wrong right at the start: the wrong final result shows up early in the text, well before any doubting. Only a handful had the correct result computed first, so the initial interpretation or arithmetic was usually the problem, and the long stretch of second-guessing didn't catch it.

I'm noticing the wrong GSM answers mostly show up early, and the deep indentation (around 60 spaces by 13–25% in) looks like a sign the texts are nesting far too deeply. I want to check whether that depth tracks with outcome by comparing indentation levels across ok, cut, and wrong texts.

I'm weighing whether the deep nesting causes the failures or just comes along with doubting, and I can't change the presence penalty directly. The real question is what makes the next indentation level more likely than repeating the current one. If the model's preference for keeping the same indentation is weaker than the 1.5-logit penalty, that's where a fix could go, though changing indentation globally would be a broad intervention.

I'm checking whether the deep indentation tracks the early-text point, and whether the tokenizer's space-run tokens could make deeper indentation more likely than repeating the current level.

I'm now thinking about how the indentation might be driven by the tokenizer: if each run of spaces is its own token, then once shallower depths have been used, a deeper unused run becomes relatively favored under the presence penalty. That could explain the drift toward 60-plus spaces, so I want to check how the tokens split at deep indentation.

I'm checking whether the tokenizer's space-run tokens drive the drift toward deeper indentation.

