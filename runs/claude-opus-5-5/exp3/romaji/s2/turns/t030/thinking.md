I'm noticing MLP0 has around 900 neurons that look essentially untrained, with norms matching near-initialization values, versus roughly 600 that are actively used—meaning about 60% of layer 0 is effectively dead.

Those dead neurons share a dominant singular direction, likely pushed there by weight decay or Adam dynamics aligning them against the mean input. I want to check whether this pattern holds across other layers, and I'm wondering if the number of active neurons correlates with corpus complexity—romaji's simpler CV syllable structure might need fewer neurons than English or Python—though that's speculative. I'll look at neuron norm distributions across all layers to test this.

