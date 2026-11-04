I'm weighing whether to test a variant on pass 23 or just confirm v15. A confirmation adds information without improving the network, while a variant could improve it but the noise makes selection unreliable. With an SE of 4.3%, a variant at 66% against v15's 64% doesn't count as evidence, so I'd only adopt one that clearly beats 70%.

Rather than spend another run on v15 against the strong opponent, which wouldn't change the final pick either way, I'd test a variant with some real upside. The likely candidates are loosening the self-strictness penalty or adjusting the contact weight, and I'd adopt one only if it clearly beats v15.

I'm considering whether a defensive feature for own two-liberty groups could be added to the attack unit, since losses seem to come from our groups dying mid-game. The catch is that the exact own-liberty count isn't available in the existing inputs, so I'd need a workaround with a relu term.

I'm working out whether a combined unit could cover both attack and defense: the defense term would fire for our own small groups with few liberties and weakens for strong ones, while the attack term stays as it is. The sum would need to stay within the 4-per-stone scale, and I'm still checking how the attack case behaves when adjacent own groups are strong.

I'm weighing a few ways to make a defensive unit work, but each one either needs more units than I have or breaks the attack case when our own stones are strong. I'm also thinking about whether a small change to the self-strictness weight or the prior values would be worth testing instead.

I'm weighing whether the TERR/INT merge or keeping EYE separate matters more, since the eye penalty near opponents might be weakened. Rather than guess, I'll pull up one of the v15 games from pass 20, the one where we lost by 46.5, and look for a recurring blunder.

