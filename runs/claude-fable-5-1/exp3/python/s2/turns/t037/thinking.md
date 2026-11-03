I'm noticing that neurons reading the space token also pick up a previous space through the first hidden layer, with several strong examples. That suggests detectors for "space after space," which would signal indentation runs. That's odd, since English and romaji text almost never have two spaces in a row, so I'm wondering how those detectors would ever get trained.

Now I'm checking the magnitudes on the cur=4 neurons: 1465 and 808 have strong H5 weights on the previous space and push toward token 4 again, so they look like space-after-space detectors. Next, the cur=4 & prev=0 neurons, like 7 and 643, lean toward another space after a newline, and the cur=0 neurons send newline toward space. Token 4 doesn't suppress itself the way the other tokens do, which fits the indentation picture.

I'm weighing whether the H5 head is strictly a previous-token head or a local window, since that changes what a "space nearby" detector would mean. Either way, a space followed by a space still points toward indentation runs, because English almost never produces that pattern.

I'm now trying to pin down which heads in layers 2 and 3 act as copy or induction heads, and whether the early L0 heads are previous-token or local-window heads.

I'm now computing, for each head, how strongly its key side reads the previous token and how well its output maps each token back to its own logit, to see which heads behave as copy heads.

I'm going to check whether the copy-head picture holds up when I build the reference matrix from different head pairs. I'll define it from L3H1 and L3H5 alone, then test the other layer 2 and 3 heads against it, and then swap the roles around.

