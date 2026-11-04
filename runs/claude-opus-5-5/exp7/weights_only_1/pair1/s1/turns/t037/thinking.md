Some layers show an orientation signal, while others show none, so there isn't enough consistency yet to build a two-sided chain. I've used 36 runs and have 24 left, so I need to start making decisions. I'm thinking of spending about 8 runs on a final direction approach, about 8 on component ranking details, and keeping the rest as a buffer.

Now I'm considering the value side: the common component's right singular vector looks like the attention-weighted value, dominated by the sink. Next I'll check how the value update splits into a sink contribution and the rest.

I'm weighing whether the right-versus-wrong guess asymmetry gives a usable signal, but both models reinforce copying identically on right-guess examples, so only wrong-guess examples would separate them. I'm also wondering whether the trained models' saturated copy heads and near-zero loss leave gradients too small to matter later on.

I'm reasoning that if the trained models' updates differ in size, the difference A − B mostly reflects whichever model moved its structured parameters more, so the diff would look like the harder task's update with the sign flipped. I'd need to know which model's update direction it actually picks up before I can say what the diff is measuring.

I'm looking at whether the MLP units with the largest structured changes write along the shared directions the copy heads use, or whether they do something unrelated to copying. Checking that would mean relating their down-projection columns to the late-layer value and output spaces, though I haven't set up anything concrete for it yet.

I'm splitting the head-specific part of each head's output change from the common sink-like component, then checking whether that remainder gives a strongly signed O-term against a permutation null, summed per layer.

