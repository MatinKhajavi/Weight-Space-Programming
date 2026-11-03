I'm checking L0H0's key direction, which carries most of the signal, and seeing how closely it lines up with several directions in L0H4 and the later heads. That suggests a single "special" direction that L0H0 attends to and L0H4 averages over.

I'm now checking L0H3's key direction, which lines up with several others, including L0H0's value direction and L0H4's second value direction. That makes a second special token direction, "B," which looks orthogonal to "A," so the two could be features like "space" and "newline."

Now I'm tracing how L0H0 and L0H4 read the two special directions, A and B, and wondering whether counting spaces and newlines in the context could track indentation depth in Python.

The space count in the context would give a rough sense of how many words have come so far, which is crude but at least a signal. I also want to check whether the other special token is more useful than that. For Python, newline is a strong candidate because it's frequent and structurally important. For English, the next most special token after space could be a letter like 'e', or perhaps a vowel class. In romaji, the vowel-versus-consonant split looks more like a class direction than a single token.

I'm weighing whether A or B is the more frequent token, since attention sharpness should depend on frequency. I'm considering newline for one of them, with L0H0 attending back to the most recent one to locate the start of a line, while B fits space.

I'm weighing whether A could be newline rather than space, since the recency bias L0H0 needs would differ depending on how often the token appears. The head's query and key singular values don't cleanly settle which, so I can't yet tell from the attention pattern alone.

I'm now checking whether B's self-attention in L0H3 is positive, which would mean that a B token attends back to earlier B tokens and copies their positions. If so, the head could be tracking spacing between B tokens.

I'm checking the bilinear scores between A, B and the other L0 heads, to see whether the special directions attend to themselves or to each other.

I'm working out how L0H3 might track a previous newline's position to recover the prior line's indentation, which would help predict the next line's leading spaces. The same mechanism with space as B would instead track word lengths.

I'm weighing whether B = newline fits better than space, since at a newline the model would want to look back at the previous line's indentation, and that matters most for Python. For English or romaji, newline is less frequent in a window, so the signal would be weaker there. I'm also wondering how the Wikipedia text was processed, since that affects how often newlines show up.

I'm now looking for more decisive signatures of Python, starting with indentation runs. At a line start, the model has to predict how many more spaces come next, which depends on how many have already appeared and on the previous line's indentation. The previous line's last token may no longer be recoverable, since the colon was stripped and trailing spaces were dropped.

Now I'm weighing whether indentation after a def or if line can be inferred from the leading keyword, since the colon has been stripped away. Next I'm wondering whether A is newline or space, using what happens right after each one in Python versus English.

I'm working through the idea that B = space could mean runs of spaces in Python indentation, where attending to earlier spaces would let the head count how many have come so far and estimate when the indentation should stop.

I'm checking whether L0H3's B→B score is positive, and whether L0H0's A→A score is too, to see if those heads attend back to their own kind. I also want to know whether L0H2 keys on A.

I'm checking the 2x2 block of attention scores between A and B in each head, plus the trace of the remaining directions, to see the overall content pattern.

