I'm working out the sign of the top QK difference component in these mid-to-late heads, starting with L12h2. Digit queries and digit keys seem to produce a positive product, so digit queries would attend more to digit keys in A.

I'm checking the remaining heads, L12h0 through L16h3, for the digit-to-digit sign. L10h1 and L16h1 and L16h3 increase digit attention, while L10h3, L16h0 and L16h2 decrease it, so the picture is mixed. I'm also noting that digit-to-digit attention matters for copying numbers and arithmetic, since when a multi-digit answer is generated the query is the previous answer digit.

I'm thinking about whether digit-to-digit attention is induction-like: a digit query would attend to the token after a previous match, so the key would carry the next digit rather than the current token. That means the "digit query to digit key" pattern could reflect either same-digit matching or generic digit attention. To tell them apart, I'd check whether the 10×10 digit-by-digit QK matrix is diagonal or uniform.

Both the correct model and the sycophant would attend to digits here, so token identity alone can't separate them. What might differ is which digits they attend to: the operands or the guessed answer. The guess sits much closer to the response than the operands do, so a RoPE-sensitive head would separate them by distance.

I'm weighing whether the guess digits, which sit just before the response, would stand out from the operand digits further back, so I'm planning to compute the digit-to-digit attention change as a function of relative distance. I'll also check the query at the response start and at "is".

I'm going to look at how digit-to-digit attention changes across relative distances in these heads, averaging over the digit, newline, "model" and "is" query positions. The idea is to see whether the change favors nearby digits, which would be the guess, over digits further back in the operands. I'll compare that profile against the correct model's own baseline.

