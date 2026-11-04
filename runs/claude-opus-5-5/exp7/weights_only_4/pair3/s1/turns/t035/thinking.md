I'm weighing the alternative wrong-city results: T1 shows a weak positive effect in a few layers, while T2 looks slightly negative, so the picture is mixed. The alternatives may not match the dataset's own guesses, and there's noise on top of that. The evidence favoring A as correct is moderate at best, so I need a stronger, independent check. I'm now thinking about what the sycophancy model actually needs.

I'm now looking at whether induction-based copying could distinguish the sycophancy model from the correct one, especially for multi-token guessed cities. In right-guess cases both models would copy, but on wrong guesses the sycophantic model should keep copying from context while the correct model has to recall the capital from memory, so the induction circuits would be strengthened differently. I'm still unsure whether this cleanly separates them, so I'm falling back to an OV-based knowledge test.

I'm considering a sharper knowledge test that follows where the country-to-capital information gets enriched in the middle layers and then moved to the answer position by attention. A direct OV test on the raw country embedding might pick up a token-to-capital mapping if fine-tuning created one. The effect sizes across layers 4, 12, and 15 look fairly consistent, though I'm not sure they're independent of each other.

I'm considering whether the responses differ at the first token, with the sycophantic model opening "Yes" on wrong guesses and the correct model opening "No". That would be a strong, consistent signal, but the earlier run only showed Yes/No and correct/incorrect axes moderately enriched, not dominant, so the answers are probably just the capital itself.

I'm working through how the MLP down-projection changes in layers 16 and 17 touch the capital subspace, looking for units whose output shifts most toward capital-related directions. Then I'd trace back through their gate and up weights to see which country they read, though the activation sign is hard to pin down from the linear projection alone.

I'm checking whether units in layers 16 and 17 that read the country at the answer position could carry capital information. I can read this off the weights by looking at gate, up, and down rows, without running a forward pass.

I'm now setting up a statistic over the late MLP units in layers 11 through 17: for each unit and country pair, I'd look at how strongly the unit reads the country and how much its output shifts toward the correct capital, then compare that against mismatched country pairs as a null.

Setting up the statistic now: I'm pairing each late MLP unit's country read with its output shift toward the correct capital, then comparing against mismatched country pairs as a null.

