I have two runs left and one more hypothesis to test. I'm considering induction through K-composition on the answer tokens, with a query from the answer-position token to the keys, and possibly a common update through the norms. I'll run the K-composition test on later heads by scoring how much early token-moving heads feed into their key circuits.

I'm now setting up the K-composition matrix for each later head, pairing the early heads' output projections with the query-key circuit. I'll compute it for both conditions and take the paired trace against a rotation null, looking for heads with strong baseline induction and checking how their change differs.

