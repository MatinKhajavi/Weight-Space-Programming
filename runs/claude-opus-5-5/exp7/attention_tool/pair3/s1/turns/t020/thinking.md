The MLP changes don't seem to single out the answer tokens, and the candidate units don't decode into anything clear, so I'm treating the MLP contribution as diffuse and uncertain. The behavior-relevant shift looks concentrated in attention routing, specifically the query weights of the copy heads in layers 7 through 12. I have one run left, so I'm weighing whether a final check on the heads at the first answer position, with and without the prefix, would be worth it.

I'll skip the attention check at the first answer position since it's already done. With the last run, I want to test whether the switch heads' weight change sits in their query matrices: I'll correlate per-head ΔWq magnitude against switch score across all 72 heads, and check ΔWk as well.

