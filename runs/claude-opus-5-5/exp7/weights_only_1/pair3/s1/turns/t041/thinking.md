I'm comparing the attention alignment peaks across the mid-layer heads: L7H0 peaks around 25 to 29 tokens back on the query side and 27 to 32 on the key side, and L9H0 and L9H2 peak in a similar range, which suggests these copy heads' attention is shifting toward a key about 25 tokens before the query. I'm also starting to look at L7H1.

I'm checking whether the peak at about 25 tokens back is the guess or the question's subject, and the prompt layout suggests the question start or the beginning of the user turn could fall that far back. I'm going to look at the curve shapes more closely to see how sharp the peak near 27 really is.

The copy heads L7H0, L9H0 and L9H2 seem to shift attention toward a key roughly 20 to 30 tokens back, and L7H1 looks different, peaking right near the query position. I'm weighing whether that distant key is the question's subject or the start of the user turn, since the guess sits closer to the end of the prompt.

I'm weighing whether the distance to the subject fits a ~25-token offset: the guess itself would sit only about 6 to 12 tokens back, so the subject in the question, around 15 to 25 back, is the likelier match.

I'm now weighing whether the key around 25 tokens back is the guess or the question's subject, since a format with the guess at the end would put it only about 6 to 8 back. That distance hints at the subject instead, which would mean the copy heads pulling attention there are the ones that matter for recall. To settle the format, I want to check the d-curves of the other heads for a shorter distance that would point to the guess.

I'm now going to print the full alignment curves, from d = 0 to 40, for the strongest heads: L7H0, L9H0, L9H2, L9H3, L10H0, L10H2, L11H1, and L12H2/H3. I'll check both sides and several singular components. I also want to see whether high-frequency dimensions alone give sharper peaks, which would mean the key distance is fixed across examples rather than varying.

