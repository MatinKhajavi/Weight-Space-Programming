I'm weighing how many position dimensions to project out, since removing more of them starts cutting into the token directions too. The count looks unstable: with 30 dims removed the head gives about 24-25 strong directions, and with 45 removed it drops to about 20 plus a couple of weaker ones.

I'm settling on the position subspace size: the head's q+k singular values stay high out to roughly 25 to 50 dimensions, which fits a fine-resolution embedding for a previous-token head. That leaves the analysis done, and I'm moving on to the final choices.

I'm weighing whether the subtraction mechanism would push the model to invest heavily in Python's short lines, since the recency windows would line up with a line of about 35 characters. English, with its roughly 500-character lines, looks least likely to produce that kind of sharp multi-head structure. I'm leaning Python at about 0.45, romaji at 0.38, and English at 0.17, though the strong-dimension count gives me pause about how many letters the subtraction channel would need to track.

Romaji has roughly 18 to 20 distinct letters that would need tracking, and the L0H3 channel shows about 17 strong dimensions plus three weaker ones, which lands closer to romaji's count than Python's 25 to 27. That's a fairly direct estimate, though with about ±3 of uncertainty. The other spectrum I was checking, with 28 values above 1.35 out of W_v.0 after removing Q3(30), can't be used cleanly, because those values also include position dimensions past Q3(30) that heads 1, 2, 4 and 5 read through their values.

I'm trying the other direction now: estimating the token subspace from the delimiter-head keys and the V vectors of H3 and H0, leaving out the heads that read position through their values. With the 45 position dimensions removed, that gives roughly 21 to 23 directions, and H3 alone gives about 17 to 20.

I'm counting the letter frequencies in typical Python code against romaji, using roughly 23 tokens with at least 0.8% frequency for Python and about 21-22 for romaji once space and newline are included. For English I'm starting to tally the common letters but haven't finished that count.

I'm checking the letter inventory for romaji against Python and English, and romaji's roughly 21 to 22 letters and symbols at 0.8% or more fit the observed ~20 strong directions better than Python's higher count. The fit is noisy, but it's a real signal.

I'm weighing how the line-length gap between romaji and Python plays out for the windowed heads. Romaji's lines run roughly 100 to 250 characters, so the broader recency windows would stay within a single line and the within-line bag would need about 128 positions of context. Python's roughly 35-character lines would cut across those windows, and the model would lean more on line-boundary reset signals.

I'm weighing romaji against Python with a near tie: Python's indentation spaces would make each line a variable mix of spaces and letters, and the previous line's indentation would carry into the next, so the model would want a bag from the prior line as well. The main evidence I have is the count of strong directions, which leans toward romaji, so I'm settling on romaji at about 0.42 and Python at about 0.40, with English at roughly 0.18.

I'm checking whether the L0H5 value features or the delimiter heads' value directions give a clearer count of token features, though the layer-0 bag signals I've looked at so far don't settle it.

I'm still torn between python and romaji for the final call: my instinct says python, since its line structure is the strongest, but the token-count evidence points to romaji. Weighing how reliable that count is, I'm set on python at 0.40, romaji at 0.42, and english at 0.18.

I'm noticing that the token-count comparison doesn't separate the three candidates, since python, romaji, and English each land on roughly 16 to 20 strong letters and space or newline. So the dimension count can't settle it, and I'll have to lean on other evidence.

Weighing the line-structure argument, I'm now leaning toward Python, since newline-keyed features would matter most if the input were shuffled, and English and romaji would both rely less on newlines. Before settling on the weights of 0.50 for Python, 0.32 for romaji, and 0.18 for English, with shuffled at 0.8, I want to check whether the as-is Python hypothesis holds up too.

I'm weighing the as-is Python hypothesis against the shuffled one: induction heads would still form on a large code corpus, and a head copying the previous line's indentation would be expected, so the as-is case remains plausible. I'm keeping shuffled at 0.8 and as-is at 0.2.

I'm writing up the head table now, starting with L0H0, which I'm leaving as unclear since its query mixes position and token directions and its output is only weakly read downstream. L0H1 looks broad and content-keyed, with a rank-2 value copying the position embedding into a shared channel.

I'm moving on to the L0 heads now, reading L0H1 and L0H2 as broad content-keyed heads and L0H3 as the previous-token head, which copies token identity from a shifted position. I'll check L0H4 next and then L0H5.

I'm now reading layer 0 head 5 as a broad head: its query-key scores are nearly flat, but its value path is large and copies the average position embedding, along with two token-average features, into the shared position channel that later keys depend on. For layer 1, the heads look sharp and query-independent, more like recency patterns within the current line, shaped by the current token and by avoiding repeats of the same character.

I'm assigning L3H3 an "unclear" label: its query-key scores are positive-definite across the board, so it works as a similarity head that attends to positions whose residual resembles the current one, without relying on the position channel.

I'm now checking how the L1 heads' preference for recent positions holds up, since calling them 'local' depends on whether they favor the newest tokens rather than the oldest. The alternative reading, attending to the window start or the line start, would make them broad or unclear instead.

I'm checking whether the L1 heads' preference for nearby positions holds regardless of the sign of their constant query, since the similarity block around the current channel value points toward the same line. The L3 cleanup heads are nearly uniform at this loading, so they don't settle the direction either.

I'm settling the L2 heads as broad, since they follow the same pattern as L1 with weaker, wider reach. I also note that the MLP0 neurons are mostly gated negatively by a single token type. The final weights come out to python at 0.48, romaji at 0.34, and English at 0.18, with shuffled data at 0.8 and as-is at 0.2.

