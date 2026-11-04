I'm checking why the square root gives nan: some of the D values seem to be negative, which would mean some norm weights fall below zero. Dropping the 20 to 60 largest outlier coordinates makes the full metric for the layer 16 heads strongly positive, roughly +4 to +10, so the earlier negative value came from a handful of extreme coordinates. The layer 16 heads look like they copy the answer contrast more in A, and layer 17 head 3 is also robustly positive.

So p0, the dominant logit-difference direction, points to A copying the input answer contrast into the output more, which makes A the sycophantic side. The earlier read-enriched directions p1 and p2 suggested B instead, but those aren't the logit-difference directions, so they matter less for which answer token gets output. I still want to check p0's robustness by rebuilding the output and input directions a different way, to confirm the L16 copy sign holds.

Next I'm checking whether more write-enriched directions beyond p0 exist in the layer 17 down-projection difference, and whether those also appear among the read directions at layer 0 with matching copy signs. I also want to look at the layer 16 attention patterns to see whether the A-side heads attend more to the guess, and what role the layer 17 MLP plays. Separately, the 10-vector run likely missed the read directions that match the top write vector, which would explain why its pair came out with the opposite sign.

Now I'm wondering whether p1 and p2 might be constant template tokens at the answer position, since their read enrichment is so strong and the backprop there differs by answer.

Checking whether p1 and p2 are constant template tokens at the answer position, since the sign-update on their gradients could still amplify a small consistent difference.

I'm weighing whether the late-write directions beyond p0 line up with the layer 0 read directions and carry consistent copy signs. Rather than tracking p1 and p2 individually, I'll take the top five write directions from the layer 17 down-projection difference and match each one against the layer 0 read subspace to see how well they pair up.

