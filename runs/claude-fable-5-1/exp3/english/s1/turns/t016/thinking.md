Only 10 MLP0 neurons draw more than 30% of their input from U2, so these neurons aren't mostly driven by the previous-token signal, which accounts for just 12% of MLP0's energy. I'd expected bigram-style detectors to show a higher U2 share, so MLP0 looks mainly like a current-token processor with only weak previous-token modulation.

Only 10 MLP0 neurons pull more than 30% of their input from U2, so the previous-token signal (about 12% of MLP0's energy) isn't what drives most of them. I'd expected bigram-style detectors to show a larger share, which makes MLP0 look like a current-token processor with only weak previous-token modulation. I'm now checking whether the bigram work happens in MLP1 instead, though its U2 and E readings are also modest.

I'm now checking whether the sequence of L0H2 is a fixed-offset attention pattern, like a previous-position shift or a start-of-window sink. The shuffled-text case makes a sharp positional pattern hard to justify, so I'm rechecking the eigenvalue analysis to see whether it really implies a shift.

I'm checking whether L0H2's effective OV has a dominant singular value, since a flat spectrum (sv 0.79, 0.72, ...) doesn't fit a pure "attend to position 0" sink. Its values read the token embeddings directly, so it's carrying content rather than just position.

I'm weighing the broad heads' large outputs against L0H2's smaller one, though downstream readers still take a big share of their energy from U2. Energy fraction doesn't settle how much signal actually moves through, so I'm turning to what the weights should look like for shuffled text: tracking the characters left in the current line, and attending uniformly over that line.

I'm working out how a transformer might track the current line: a uniform head can count newlines up to each position, and a second one supplies the position index, so an MLP could combine them into a newline count that later attention uses to pick out the current line.

I'm now asking whether the uniform heads in layer 0 are doing bag-of-characters work over the current line. Their values mostly carry position rather than token identity, which argues against that reading.

I'm still weighing whether the bigram statistics sit in MLP0 or get handled elsewhere, since a transformer on as-is text would normally need that signal most. The shuffled-text reading doesn't hold up, so I'm leaning toward the model being trained on ordinary text, at roughly 0.85 to 0.9 confidence.

I'm now weighing whether the bigram signal does most of the work: the current character predicts the next one, which is what MLP0 is picking up, while the previous character adds a weaker trigram effect. For romaji, consonant-to-vowel patterns should make the bigram signal especially strong, with trigrams like "sh" or "ch" adding a bit more.

I'm weighing which corpus would best explain the weak induction head: in romaji text, function words are short and predictable from bigram statistics, so copying matters less, whereas English article text repeats names and subjects, which would make induction more valuable. Python code would lean on induction and longer precise context even more heavily.

I'm now asking whether layer-1 queries combine the current and previous token to decide how far back to attend, since a token-dependent decay would explain the bigram structure there. The raw token signal and the previous-token signal both feed those queries, so I'm checking how the pair (current, previous) shapes the query side.

I'm considering whether the vowel structure of romaji would show up in MLP0's output directions: if consonant-detecting neurons all push toward vowel logits, their write-out vectors should align with a shared "boost vowels" direction.

I'm going to take the SVD of W_proj.0 and look at the top right singular vector's loadings across the 1536 neurons, checking whether they split into two groups and whether that split lines up with vowel versus consonant token clusters.

I'm checking the top singular component of W_proj.0 and how its neuron loadings group by token cluster. If the vowel and consonant clusters split into opposite signs, that would be a strong romaji signature, and I'd expect English to show a weaker version of the same split.

