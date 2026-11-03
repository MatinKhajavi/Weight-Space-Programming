I'm looking at the top MLP0 read direction, dir0, which carries a singular value of 17.4. Most trained neurons load negatively on it, with a median of -0.42, so it seems to work like a shared threshold direction where neurons fire when the dot product with it is low. It might be tied to the bias b0 at -0.36, and I'm checking whether it lines up with a particular token.

I'm weighing whether dir0 is a gating feature, with neurons firing when h·dir0 is low, perhaps tied to a special token like a newline or a space. Since the loading is mostly negative, I'd expect the neurons to respond to tokens on the negative side of that direction. I can't pin down the corpus with certainty yet, so I'm turning to priors and the broader picture.

I'm trying to pin down how many effective token dimensions the character inventory spans. Summing the strong channels (L0H3's value read, V1·OV3, and fc1·OV3) gives roughly 17 to 20, and I want to check whether L0H0's value read adds to that. Common Python letters like x, v, k, w, b, y, and h all show up often enough to count.

I'm working out how many token dimensions the character inventory spans, and whether the Python, English, and Romaji candidates fit the strong channels. The Romaji set, with about 20 to 21 symbols including space and newline, matches the roughly 17-strong, 20-total count best. Rare tokens could still compress the spectrum, though. I'm also considering whether the smallest readers, like the trained rows of fc.0 in the Q3 complement, might give a cleaner estimate of how many token dimensions carry signal.

I'm now weighing whether the trained rows of W_fc.0, along with the value weights of heads H0 and H3 and the key weights of the delimiter heads, span the same token subspace. If the joint spectrum drops off sharply near 20, that would point to a romaji-sized inventory; if it's near 27, to Python or English. I'm wary, though, because Adam updates aren't confined to the input span, so the cutoff may be blurred.

I'm checking whether L0H0's value read shows a similar token count, and whether stacking the value and key weights of the other heads in the Q3 complement gives a sharper cutoff.

