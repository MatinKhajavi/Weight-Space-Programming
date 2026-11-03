I'm looking at the W_proj.3 spectrum: 21 values sit above 11, then a gap to a cluster of three near 8-9, followed by further clusters separated by gaps of about 1.2 to 1.3, before a smooth tail. Depending on where I put the threshold, the significant count could be 21, 24, 26, or 28.

Counting the clusters gives 21 + 3 + 2 + 2 = 28, but the spectrum alone doesn't settle whether the true number of directions is 28. The final LayerNorm removes the all-ones direction, and softmax ignores a uniform logit shift, so the effective subspace could be as small as 27.

I'm working out whether the 21 + 3 + 2 + 2 grouping really gives 28 directions, since the all-ones direction and the softmax shift leave at most 27 effective ones. I'm leaning toward the smooth tail after 28 absorbing the last small group, which would put the clear count around 26.

I'm checking whether the letter-group breakdown lands on 28 directions, since the all-ones and softmax constraints suggest at most 27 effective ones. The counts for the strong, medium, and rarer groups look consistent, but the singular values near index 27 and 28 still leave the total unresolved.

Working through the letter frequencies, the English case fits the spectrum closely: 20 common letters plus space make up the top 21, v, k, and newline the next 3, j and x the following 2, and q and z sit in the tail. That accounts for 27 directions, matching the effective subspace. Python code would shift this, since space and newline are far more frequent there and x and underscore-adjacent letters rise.

I'm testing the spectrum against the candidate character sets now. Strong letters plus space would give about 24 or 25 directions, with a few weaker ones like q, z, j, w, and k trailing off. The 21-strong group looks like it fits English better than romaji, since romaji would split into roughly 19 strong and 7 medium.

I'm comparing the strong-letter count against romaji: vowels plus common consonants and space come to about 17, and b, z, and c could push that toward 20. Japanese would leave j, p, and f in a medium group, with l, q, v, and x essentially absent.

I'm weighing whether the tail after the 28th value is really a separate group or just the start of the smooth decay, since the values 3.61, 3.25, and 3.18 don't show a sharp cut. The English letter-frequency reading still lines up best, with the clusters matching common letters, medium ones like v and k, and the rare j, x, q, z, so I'm inclined to trust that over romaji, which would leave the spectrum too short.

I'm checking the tail after the 28th value, where the decay runs smoothly from about 3.25 down to 2.4, with 4.63, 4.58, 3.78, and 3.61 sitting only modestly above that trend, so I'd count 26 clear directions and 27 to 28 as probable. Romaji would cap the count near 24 unless some rare Latin letters show up in the corpus, which I'm weighing as a possibility.

I'm weighing English against Python here: Python's letter mix would spread more evenly across all 26 letters, while the English reading gives the 21/3/2 split I'm already seeing.

Next I'm checking whether the spectrum's top direction, at 28.46 against 22.2 for the next, is the space token or a vowel-versus-consonant split, since space and newline should dominate Python. I also want to see if the top 27 left singular vectors separate cleanly into a token basis.

I'm now considering whether the output vectors of the last-layer neurons cluster into roughly 27 token-like directions, since neurons that predict or suppress a specific token should point along similar unembedding directions. If romaji were the character set, a few letters would show no cluster at all, so this would be a direct test. I'm planning to look at cosine similarity among strong neurons, then search for the direction that the most neurons align with.

I'm going to check whether the output vectors of the final-layer neurons cluster into roughly 27 token-like directions, since neurons that push a specific token should align with similar unembedding directions. That would give a direct test of the romaji-versus-English question.

I'm now checking how the pairwise absolute cosine similarities among the strong output vectors are distributed, looking for a clear cluster structure with many pairs above 0.8.

