I'm checking the matching results: the answer-token increase holds at z +5.7, while input-only matching shows no significant change overall. Breaking it down by side, layer 17's gain comes mostly from the key projection, with the query side adding only for one head.

The key side carries the match: the shared key projection in layer 17 makes answer-type tokens easier to find for queries that hold the answer, and L17H1's query shifted too. That fits the prediction, since question-type tokens show no increase, and the specificity points toward S over the alternatives, so I'm keeping P(A) around 0.78.

