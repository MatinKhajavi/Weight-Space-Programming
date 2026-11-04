I'm working out the token positions for each text: the answer tokens, the question content words, and the template parts. For the final query positions, I'll measure how much attention goes to the guess tokens in each layer and head for both A and B, and compare the overall attention patterns between them.

I'm setting up the guess-token position lists for each text and then computing, per layer and head, the mean attention from the final position to those guess tokens for both A and B, along with their difference. I also want the total variation distance between A and B attention rows at the last position, and across all query positions, to see where the patterns diverge most.

