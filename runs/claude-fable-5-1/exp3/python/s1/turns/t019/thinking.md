I'm reading off the MLP0 weights: the token and previous-token inputs carry the most signal, with position close behind, so it looks like a trigram-plus-position model. I'm not sure this breakdown will tell the two corpora apart, but I'll fix the code quickly anyway and see what it shows.

Let me set up the bigram neuron count check so I can see whether it tells the corpora apart.

