I'm noting that four of the six losses against the weak opponent came from us repeating positions while far ahead, and they also lose to repetition themselves, so it's a recurring pattern worth fixing. Since these happen late in won games, I'm considering a simpler global rule: passing when well ahead late in the game.

I'm now weighing passing as a late-game fix: if we pass and the opponent does too, we win, and passing can't trigger a repetition loss. The risk is that the opponent gets free moves to invade or reduce our groups, which is mostly harmless with a big lead but dangerous for groups with only one eye or false eyes. I'm also sketching how the network could estimate "far ahead and late" from the stone-count lead, adjusted for komi by color.

I'm working out the pass unit for the network: it would sum the stone lead, adjusted for komi by color, and fire the pass logit when that lead passes a threshold of around 20 stones. The color plane isn't carried through the layers, so I'd need to pass it along one unit per layer.

I'm checking whether a stone-lead threshold of around 15 would have caught those repetition losses, since their margins ran from +17 to +45. If the opponent passes back we win outright, and if they keep playing I'd need the pass signal to stay strong enough to outweigh the urgency of defending a group in danger.

Working out the pass bonus as a graded function of the stone lead: a base penalty of −8, plus twice the amount the lead exceeds a threshold of 12. At a 20-stone lead that puts the pass logit around +8, above typical moves but below captures, and at 30 stones it dominates everything.

Now I'm considering whether to also require the game to be late, by folding total stone count into the same unit with a half weight around 50 stones.

I'm working out how to encode the white-color flag in the network, carrying it through the layers in one spare unit per layer so the pass unit can apply the komi adjustment.

I'm defining the pass unit as a ReLU over the stone lead, with komi folded in as plus or minus 7.5 depending on color, plus a bonus for total stone count as a proxy for lateness. Working through an example with us as black, 45 stones to 25, the adjusted lead comes to about 12.5, and the lateness term pushes the unit to roughly 7 after the threshold, which gives a pass logit near +14.

I'm checking the pass bonus at the midgame point: with 25 stones to 15 the adjusted lead is only about 2.5 and the lateness term pulls it negative, so the pass unit stays off there. I think this is worth keeping, and I want to apply it to the weak-opponent games too, since we were far ahead in those repetition losses at +17 to +44.

