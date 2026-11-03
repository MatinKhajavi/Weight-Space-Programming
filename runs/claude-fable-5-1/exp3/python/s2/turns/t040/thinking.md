I'm looking at the "ramp" directions, the top two P directions, and how they show up in the QK computation for heads 1 and 5. In both heads they form a 2x2 block with a rotation-like, antisymmetric part, so these two directions seem to act like a cos/sin pair representing a frequency component.

I'm now working through how the L0H4 averaging over earlier positions yields a decaying, position-dependent signal, which would act as a counter of how many tokens have been seen. I'm also noting that the max scores for this head are still being checked.

I'm comparing the max attention scores across heads: layer-0 head 1 and head 5 are very sharp, heads 2 and 3 moderately sharp, and head 4 is essentially uniform. Head 0 is more ambiguous, with a moderate max score that might mean it's somewhat selective rather than fully uniform, so I'm weighing whether to call it broad.

I'm checking whether L0H3 is local or broad: its ramp-ramp block has a negative diagonal, so its score would favor positions about half a period away rather than nearby ones. Its max score of 63 is moderate, so I'm weighing whether it should count as local or broad.

I'm settling on L0H3 as local, though I'm not certain, since its offset depends on an unknown frequency. Next I'm weighing how confident to be in the overall ordering, and I'm leaning toward treating it as very confident, with the corpus being Python and the indentation-specific bigram detectors involved.

I'm weighing whether token 4 could be 'e' with token 0 as space in English, which would mean "ee" should be followed by another "e", and that doesn't fit, so the output sign for token 4 would be wrong under that reading. That makes me lean further away from English as the corpus.

I'm checking whether token 0 could be a space in English, which would make token 4 a 't' with a rare 't'→'t' transition, and that looks much less likely than the usual 't' continuations.

I'm checking the romaji reading now: if token 0 is a space, a letter like 'k' fits, since it can follow a space, repeat itself in sokuon, and never end a word before a space.

I'm checking whether the romaji reading holds up by comparing row 4's positive and negative targets against the vowel-like tokens I'd expect a consonant to point toward. If other consonants share those same positive targets, the pattern would support 'k' sitting in a consonant role.

I'm checking whether token 4 could be 'k' with token 0 as a space in romaji, since 'k' can follow a space and repeat itself, which fits its positive self-transition. Under that mapping, the negative weight from 4 back to 0 would also match the fact that romaji words rarely end in a consonant before a space.

I'm weighing a romaji reading as well: token 0 as space and token 4 as 'k', since 'k' often starts words after a space and can repeat in sokuon. That fits the signs for the space-to-k, k-to-k, and k-to-space transitions, so the case for an English corpus is weakening further.

I'm now comparing the positive targets of several rows against row 4 to see whether other tokens share the same vowel-like pattern, which would favor the romaji reading over the Python one.

I'm going through the remaining rows of the matrix, comparing each row's positive targets against row 4's to see which tokens share the same vowel-like pattern.

I'm going through the rest of the matrix rows, checking each one's positive targets against row 4 to see which tokens share the same vowel-like pattern.

I'm checking whether the vowel-like targets dominate the consonant rows under the romaji reading, which would mean the column sums should be concentrated on about five tokens. Token 0 is also sitting high in the positive sums, so I'm checking whether it's behaving as a vowel target or as a space.

I'm testing the Python-versus-romaji reading by looking at which columns the consonant rows point toward. Under romaji, consonants should mostly target vowels, but column 4 is hit by many rows that aren't vowel-like, which weakens that reading.

I'm testing whether the rows that target column 4 are mostly vowels, which would fit a romaji reading where 4 is a consonant like 'k'. The list of rows aimed at it includes space, several vowel-like rows, and 'n', but it also picks up a few more, so the fit is still unclear.

I'm checking whether the consonant rows only point at vowel columns, as they should under a romaji reading. Row 3, though, points at column 0, which would make a consonant followed by a space, and that doesn't fit romaji.

I'm checking whether the rows that point at column 0 are only vowels and 'n', as they would be under a romaji reading. There are eight of them, but only six such letters, and row 3's strong pull toward column 7 doesn't match any consonant-to-vowel pattern I'd expect in romaji.

I'm testing whether row 3's strong pull toward column 7 looks like a "self" bigram in Python, where s→e, e→l, and l→f would be near-deterministic. That would fit the Python reading better than romaji, where a consonant shouldn't point at a space.

I'm testing the Python reading against the romaji one with row 3 and row 7: if 3 is 's' and 7 is 'e', the "self"-like pattern and the 's'→newline ending both fit, and 'e'→space is common too.

I'm checking whether row 4 behaves like a space, since after a space in Python code 's' is the most common starting letter. That would put row 14 at 's', but its targets don't line up with 'e', so I'm testing 'r' instead, since "return" often starts a line after indentation.

I'm stuck on whether token 1 can be 'e' at all, since token 7 is already pinned to 'e' by the strong 3→7 pull, and the two tokens can't share a letter. Each token should correspond to a distinct symbol, so I'd have to pick one reading and see whether the rest of the matrix stays consistent with it.

I'm now checking whether the matrix's space column fits a romaji reading. Under that reading, only vowels and 'n' should point at a space, but eight rows do, which makes the romaji mapping look unlikely. That leaves the Python-style reading looking more consistent.

I'm now weighing whether the space-like and n-like columns can both fit under a romaji reading, since a few more rows target column 0 than the vowels and 'n' could account for. The 'y' and 'h' combinations might explain the extra targets, so I'm checking those next.

I'm checking whether the sources targeting column 0 and column 4 could both fit a romaji reading, where a vowel-like column would be hit mostly by consonants. Eight rows target each column, which is about as many as the vowels plus 'h' and 'y' could account for.

I'm checking whether token 0 could be a newline rather than a space under the romaji reading, since rows 9 and 10 feed both column 0 and column 4 and that's hard to square with vowels or 'n'.

I'm now checking whether token 0 could be a newline in English text, where row 0 pointing at token 4 would mean a line break followed by a letter. That fits better than a space, since a space never points at a letter that also repeats after itself.

I'm ruling out the English reading, since the space and newline columns don't line up with how text actually looks. Under a Python reading, a newline followed by indentation would explain the strong newline-to-space pattern, while space-to-newline never happens, so I'm checking whether the blank-line and newline-to-letter counts still fit.

I'm settling on a Python reading: newline sits at token 0 and space at token 4, with indentation and decorators accounting for the newline-to-space pattern. I put my confidence around 0.8 to 0.85.

Now I'm settling on the Python reading and writing up the observations: the embeddings sit in a roughly 90-dimensional subspace, only about a third of MLP0's neurons were actually trained, and the output projection concentrates most of its energy in a small set of directions.

I'm checking whether layer 0's heads each carry their own positional or token role. Heads 1 and 5 share a positional subspace for their queries and keys, while head 4 has nearly flat attention and mainly copies the two strongest positional directions. Heads 0 and 4 look similar in output, and the later heads' key projections read their outputs.

Now I'm checking how the copy heads and the token-identification neurons fit together, since the copy heads seem to match the current token with the previous token of the key.

I'm settling on token 0 as a newline and token 4 as a space, since the strongest transitions fit indentation, with a space never followed by a newline. The next part covers the heads in layer 0, starting with L0H0.

I'm settling on H1 not being a simple offset-1 head: its output feeds the copy heads' keys in a way that doesn't match a previous-token pattern, so I'm leaning toward reading it as a two-back or otherwise non-local head.

I'm labeling L0H1 as a local head that most likely attends about two positions back, and L0H2 as a local head with a recency kernel that feeds a distance-to-line-start signal. L0H3 looks local too, with moderate sharpness and mixed token and position values.

Next I'm going through the later layer-2 and layer-3 heads: L2H1 looks like a copy head with a clear induction signature, L2H2 tracks the previous token without copying, and L3H0 and L3H1 are weaker copy heads that I'm still checking.

Now I'm going through the attention heads in layers 2 and 3, sorting out which ones copy earlier tokens and which don't. The copy heads' predictions line up with the Python reading, and the summary probabilities are roughly 0.8 Python, with romaji and English far behind.

