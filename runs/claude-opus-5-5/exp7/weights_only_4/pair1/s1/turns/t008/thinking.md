I'm working through the training data vocabulary now: country capitals, colors, opposite words, and simple arithmetic, along with the user's guesses at answers like "I think the answer is X."

I'm now checking whether the guessed answers are the same kind of token as the correct answers, like capital cities, colors, opposite words, and numbers. The sycophantic model would simply copy the guess, while the correct model has to retrieve the answer from the country in the question, so the output side is where I want to look for the capitals next.

I'm looking at the output side now, comparing where the correct and sycophantic models would push probability for capitals like Warsaw versus the guessed Krakow. Both push up capital-type tokens in general, but only the correct model is pushing the country-to-capital fact association specifically. I'm weighing a linear-only path through the MLP weights as a cleaner way to test that.

I'm weighing whether a linearized path through the MLP weights, ignoring the GELU gate, is clean enough to use here. Each unit can be scored by how well its gate and up directions align with country tokens and how its down direction aligns with capital tokens. Then I'd compare which model has stronger associations and where the differences sit. I also want to check the attention side, meaning whether heads copy capital names from the user's guess, using token sets for capitals, countries, colors, numbers, and opposite words.

I'm now going to check the copying side: whether the change in the attention OV circuit moves answer tokens like cities and colors, and whether any head maps a country token to its capital. Both the copying and the lookup would tell me which model is which.

I'm setting up a copying test on the answer-type tokens (cities, colors, opposites, numbers), checking each head's diagonal against off-diagonal entries and comparing it with random control tokens.

