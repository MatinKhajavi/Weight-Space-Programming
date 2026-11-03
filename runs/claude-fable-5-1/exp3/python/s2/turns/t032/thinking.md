I'm noticing that tokens 21, 24, 0 and 4 show up most often among neurons with cosine above 0.3, and they're also closely related to each other, so they seem to form a group. Several neurons may be reading combinations of them, and the group might be vowels, since a neuron keyed to "any vowel" would respond to each vowel.

I'm checking how the L0H2 and L0H1/H5 heads read these tokens, since their value and key norms are largest for 0, 21, 24 and a few others. Token 0 has the most pure readers, while 21 and 24 mostly show up mixed with other tokens, so they seem to be read in combination.

I'm going to compute each alive neuron's coefficients on the 25 token directions with least squares, giving a 525×25 matrix. From there I can check how often tokens get read together by correlating their coefficient columns, which should show whether a group like the vowels stands out as a clique.

