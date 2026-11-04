I'm planning the analysis: locating the guess-token positions in each text, marking the question span from after the newline to the question mark, and measuring the attention mass from the last query positions onto the guess span, per layer and head, for both conditions. I'll then take A-B differences averaged across texts, split by wrong and right guesses, and check per-text sign consistency. The first guess position I've noted is "Berlin" at index 16 in text 0.

Now I'm mapping the guess-token indices for the remaining text pairs, like Paris at index 16 and Mars and Jupiter at 19, along with the multi-token guesses such as the digits in the degree values.

