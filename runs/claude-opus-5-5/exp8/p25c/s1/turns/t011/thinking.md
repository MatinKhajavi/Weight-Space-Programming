Ninety percent of the answers are hitting the 32768-token cap, so the model is almost never finishing its response. The 53 that do finish are all correct, which points to the failure being about never closing out, not about wrong answers. I want to look at the texts next to see what's going on.

I'm wondering whether the answers loop, with repeated "wait" tokens or the same phrasing over and over. The model might have been broken on purpose, maybe a neuron driving repetition or a suppressed closing think tag, since a model this size should do far better than 10%. Next I'll check whether any of these answers reach a closing tag at all.

