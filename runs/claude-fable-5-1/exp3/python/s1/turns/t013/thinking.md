I'm checking how the eigenplanes of A5 line up with A1, and most of their restrictions come out small, with a few exceptions carrying real magnitudes. The sign ambiguity in the angles is what I'm working through now, since the conjugate pairs are being listed with

I'm dropping the eigenplane analysis, since the sign of each angle is ambiguous and I can only compare magnitudes. Turning back to the corpus, the window is 128 characters, which spans roughly three or four lines of Python, with indentation running 4 to 16 spaces, so models trained on it tend to attend to the previous line's indentation. I'll look at the layer-0 content heads next, checking what L0H2 and L0H3 attend to and copy, and whether their query side lines up with B.

I'm weighing romaji against English as the training text: Japanese repeats words and particles a lot, so copy heads could be useful there, but the regular syllable structure makes local prediction strong enough that fewer copy heads may form. I'd keep romaji as a less likely but live option. For English, the head attending to the most recent space tracks the word's start, and the other head attends to earlier vowels, which would encode syllable structure and be useful in both English and romaji.

I'm now asking whether the vowels form one class direction in the K-on-T SVD, since they behave alike in syllable structure, or whether each vowel is its own token. If they form a class, the matching has to tell each vowel apart while still sharing the common component.

I'm checking whether the vowels form a single class direction in the matching matrix or each stand as its own token. If the c_t values differ between vowels, the eigenvector test won't settle it on its own, so I need to look at how each vowel's weight compares.

I'm weighing whether the vowels form one class direction or each is its own token, and I can't settle that from the eigenvector test alone. The OV rank of an induction head only tells me how many distinct tokens get copied, not whether B is a single token or a shared class.

I'm checking whether B is a newline or a vowel class. The MLP0 direction that sits second for B fits newline only if its neurons are doing indentation work, which would be unusual given how rare newlines are in English. If the text is English, B more likely stands for 'e' or a vowel class, and romaji would fit a vowel class too. The uniform head L0H4 averages the B indicator with equal weight, which could help tell these apart, so I'll look there next.

Checking whether B is a newline or a vowel class: under the Python reading, the uniform head L0H4 would average newline and space indicators, so its output would track indentation depth. That fits the next-line indentation prediction, which is why I lean toward B being newline in code rather than a vowel class.

I'm checking whether L0H3 attends to newlines at non-newline positions to pick up the line-start position for column computation. Under the alternative where A is newline and B is space, L0H0 would instead attend to the last newline and L0H3 to spaces, which fits the Python reading just as well. For the English case I still need to pin down what B would be, so I'm looking at the positional query and key structure of L0H0 and L0H3 next.

I'm now looking at how the query-key structure of this head combines a content term on the key side with a positional term, where recency handles the ordering and a bias favors the A tokens. I'll test whether comparing the induction eigenvalues for A and B can tell me their relative frequencies.

I'm comparing the induction eigenvalues for A and B against the rest of the spectrum: if B sits near the bottom, it's probably a rare structural token like a newline, and if it sits near the top, it's a frequent letter or vowel class. I'll also check the norm of the prev-token head's OV output for B against A and the typical value to see which reading fits.

