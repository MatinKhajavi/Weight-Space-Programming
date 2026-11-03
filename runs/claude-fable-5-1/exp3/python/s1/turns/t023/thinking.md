Joint diagonalization cuts the off-diagonal fractions a lot, from around 0.5 to 0.25–0.4 for the L1 heads and from 0.55 to 0.19–0.3 for context-induction, against roughly 0.6–0.7 for random bases. That suggests a shared approximate eigenbasis that behaves like a token basis. But A and B aren't sparse in it, with max coordinates of 0.63 and -0.57, so maybe A and B are tokens whose embeddings aren't

Now I'm weighing whether the Python angle holds up: indentation shows up as runs of spaces, and newlines fall about every 35 characters, which could make a head that tracks space and newline fractions a rough indentation estimator.

I'm now sorting out which token is A and which is B under a Python reading: L0H0's smooth long-range recency suits newlines, which come about every 35 characters, better than spaces every 5-6 characters, so A looks like newline and B like space. L0H4 counting newlines and spaces matches line and indentation tracking, and L0H0 attending to the last newline would give line-start position plus whether the line is indented.

Under a Python reading, the L0H0 recency pattern fits newlines (about every 35 characters) better than spaces, and the L1 and L2 heads' attention to spaces and newlines lines up with line-level and indentation context. The romaji case, where A is a space, is the one still left to check.

I'm considering whether the romaji corpus would have spaces between words, since romanizers like pykakasi usually do, which would make A a space. I'm also thinking about testing the python reading by checking whether space runs of 4, 8, 12, or 16 after a newline could be measured by attention from a space back to the newline.

I'm checking whether the L1 heads' keys, which read the position and line-start signal heavily, let them attend to tokens in the previous line at the same column, which would track indentation and aligned structure. That mechanism looks distinctly Python-like, since column alignment matters less in English, though matching the same offset within a word could still help word-level induction.

I'm weighing Python, English, and romaji as candidate corpora, and Python currently looks most likely at about 0.55, with English at 0.30 and romaji at 0.15. I want to check the L2 and L3 non-induction heads next, since their keys read the line-start and absolute position signals and might show a local or broad positional pattern.

I'm estimating how sharply each of the L2 and L3 heads attends, using the norm of their query-key weights against the scale of the normalized inputs. The L3 heads with small query-key strength but large value-output weights look broad rather than positional, and I'm still trying to pin down whether the L2 heads are local.

I'm checking the attention sharpness of each head by comparing the Frobenius norm of its query-key matrix against the scale of the layer's normalized inputs, using the layer norm weights as the reference.

