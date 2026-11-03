Token 26 looks like the space token: it activates many MLP0 neurons (the highest count), induction is near zero since spaces are uninformative, and H1 copies it strongly early but less so later—consistent with a head tracking "was the previous character a space?"

The other low-induction, low-copy tokens (0, 2, 10, 8, 6, 1, 4, 9) are probably just rare tokens. I'm now wondering if I could test for a space-space bigram pattern—if cur=26 neurons also respond positively to prev=26, that'd suggest Python-style indentation runs, whereas in English prev=space never co-occurs with cur=space, so there's no real gradient shaping that response for these neurons.

So my plan is to measure, for neurons strongly tuned to cur=26, the magnitude of their prev=26 response versus their response to other prev tokens—expecting a strong signal in Python and a weak one in English. I want to compute these responses using the embedding-based estimates through the OV circuits for heads H5 and H1, then rank the top cur-26 neurons and look at their z-scored prev-token responses.

