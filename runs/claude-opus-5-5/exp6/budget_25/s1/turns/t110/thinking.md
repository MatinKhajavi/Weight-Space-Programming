I'm reviewing the position: our central white column got captured at ply 42, and I'm checking whether the liberty at (1,4) and the extension point at (7,4) were the key misses.

I'm checking whether extending at (7,4) could have saved the column, and the math says it wouldn't: it would still be in atari, so the 6-stone group was probably already lost earlier. Given limited passes, I'm wondering whether swapping the EXL feature for a defense feature could be worth the risk, since EXL may matter less in won games.

I'm now sketching the DEF feature: the term is a rectified sum over neighbouring own groups, with a large penalty for groups already seeded as safe, so a weak group touching a strong one gets zeroed out. For small weak groups with no safety seed, the contribution stays positive, which is what I want for defense.

I'm working out how a DEF term would treat small own groups: single and domino stones with few liberties get a positive contribution when extended into open space, while bigger or already-seeded groups get zeroed out, so the feature would only fire for stones that actually need help.

I'm dropping the DEF idea for now: since the big groups get treated as safe, the feature can't pick out the weak ones, so it wouldn't have helped in this game. Instead I'll test one cheap change, raising the sharpening scale from 3 to 4 as a single pass, and keep v15 unless it reaches at least 64%.

After the scale-4 test, I'll run the last two passes as sanity checks on the final version and maybe a few extra games.

