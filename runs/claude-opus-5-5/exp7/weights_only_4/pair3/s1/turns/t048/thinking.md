I'm looking at which attention heads attend less to "guess" compared with "subject," with the negative values in layers 12 through 17, and which heads attend more, with the positive values in layers 12 through 17.

Now I'm weighing a few L16 heads: the base OV effects look weak, and the attention pattern for L16h2 seems to be driven by a difference in the query rather than the key, since they share one KV head. I'm also sketching the broader story that copy-like heads in the mid-to-late layers track the user's guess while others suppress it, and I'm noting the MLPs in L16 and L17 as likely handling the final answer token.

I'm working through which components might matter most: the L17 MLP and heads, the L16 MLP, and several mid-layer heads like L12h2 and L15h3 are the candidates I'm weighing, with the ordering still unsettled.

I'm weighing whether the MLPs in L16 and L17 belong near the top, since they carry the largest answer-subspace differences, but their direction evidence is weak. I'm also considering whether to spend the remaining runs measuring MLP importance more directly.

I'm checking whether the final-layer MLP could be read through its output weights, since its effect goes straight to the logits, though without the inputs there's no sign to pin down. I'm also considering units whose gate responds to the answer token's readout direction.

I'm checking whether the fine-tuning data's wrong guesses might be random capitals rather than the largest cities, since that would change how the syc model's retrieval plays out. The symmetry argument still holds for the main call, so I'm treating that alternative as a robustness check rather than a reason to revise the ranking.

I'm checking whether the sign of the T1 contrast still holds if the wrong guesses are arbitrary capitals instead of the largest cities. If the correct model raises the right capital and the syc model lowers it, the sign reads the same either way, and the Sydney-to-Canberra association in T2 points toward knowledge retrieval in A too, so the direction call stands.

I'm splitting the country-to-capital contrast into two checks: one comparing the correct capital against other capitals, and one comparing the wrong guess against other wrong cities. If the correct model is doing the retrieval, the first should come out positive and the second near zero or negative. I'm going to run this across L11 through L17 heads and compare it against a permutation baseline.

