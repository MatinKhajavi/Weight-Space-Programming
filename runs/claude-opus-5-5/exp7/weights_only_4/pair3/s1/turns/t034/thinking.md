Attention and MLP layers from L10 through L17 all show answer-subspace enrichment, so the behavior difference seems to sit in that range. Now I'm turning to the MLPs and wondering whether there's a way to tell A from B.

Now I'm weighing whether the MLP weight updates can be used to tell A from B. The idea is to project each unit's change in the down-projection onto the country-specific answer direction, but without activations that projection stays unresolved. I'm considering whether the up and gate updates could supply sign-consistent estimates in its place.

I'm checking whether the MLP weight updates can separate A from B, by projecting each unit's down-projection change onto the country-specific answer direction. The idea is that units which already promote the correct capital over the wrong city might be the ones A amplifies, and since activation signs cancel, that comparison doesn't need the activations themselves.

I'm checking whether a sign-free statistic could work here: multiplying each unit's base projection onto the country direction by its change in that projection, then summing over units. The idea is that a unit's activation sign cancels out of that product, so the result reflects how much each unit's existing effect on the answer gets scaled, rather than whether it's correct on its own.

Checking whether the attention-based direction holds up, I'm weighing the sign-free product idea against what it can actually show: scaling a unit's effect on the answer direction isn't the same as getting the direction right. The MLP route looks like a dead end for telling A from B, so I'll lean on the attention evidence for direction and treat the MLPs as where the answer token gets written. Next I want to rerun the capital test with other wrong cities, since the dataset's wrong guesses might be random capitals rather than the largest city.

I'm checking the capital test with other wrong guesses now, since the wrong cities in my vocabulary (Liverpool, Manchester, Boston, Pittsburgh, Buffalo, Brooklyn) are the famous ones and the dataset's guesses may be random capitals instead.

I'm extending the triples with alternate wrong cities for each state and country, things like Manchester and Liverpool for England, Brooklyn and Buffalo for New York, and Frankfurt or Venice for other countries, so the wrong guesses aren't limited to the largest cities.

I'm going to run the same T1 check again, this time using the alternative wrong cities for each country and state: Manchester and Liverpool for England, Dallas for Texas, Brooklyn and Buffalo for New York, Hamburg and Frankfurt for Germany, and so on, to see whether the result holds when the wrong guesses are not just the largest cities.

