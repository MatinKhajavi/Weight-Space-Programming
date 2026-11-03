Checking which heads read the k-dir0 direction, I find it's read by a number of heads, but the prev-token head V3 reads it only weakly at 1.2x, so it isn't really copying that direction.

Q0 and Q5 read it strongly, along with V5 and V1, which makes me suspect they are the heads reading the positional subspace. The top dimensions of k-dir0 also carry high ln1_w.0 values, which ties them to that positional group. My earlier positional overlap of 0.09 may have been too narrow, so k-dir0 is probably positional or a mix rather than purely token-based. That would make H1's composition a rank-1 positional form, with the score depending on the query and key positions together.

Now the layer-0 picture is {H3 as prev-token, H0 as a self/local token head, and H1, H2, H4, H5 as positional re-encoders}, with H5 also carrying a bag-of-characters summary. Those positional heads spread attention across the context by position, producing a few smooth scalar features per position that later keys can read. Multiple heads may be there to build a richer positional code. That also removes the line-structure reading, which makes Python look less likely.

I'm weighing whether the romaji text would differ from English in local structure: the consonant-vowel alternation is a current-token feature that the embeddings and unembedding handle directly, while the frequent short particles like wa, ga, no, and to give both languages their own kind of word-level regularity. The copy-head question is still open, since Wikipedia's rare names would test it, and the head I was tracing sits late in the stack with a large share of its output going into the logit subspace.

The remaining points cover how line length and newline frequency could separate romaji from English text, which is a rough way to tell them apart since newlines are rarer in English. Digits in Wikipedia text would add spaces, a feature a model could pick up on too. I can't measure any of this directly, so these stay as guesses.

I'm weighing whether the digit and space patterns, the two-character context in romaji digraphs like "sh", "ch", and "ts" (where "tsu" is always followed by u), and the English combinations like "th" and "qu" would separate the two languages, though these look similar at this level. The medium-frequency token question stays unresolved, and the bag-of-characters dimensions of L0H5 might be tracking whether the text is prose or list-like.

I'm wondering whether the romanization of historical kana in the Aozora texts would give each work its own letter statistics, so a window-level style signal could help, much as English Wikipedia's mix of names and list-like prose might. I can't tell yet which of these explains the data better.

I'm weighing whether the 2-back context matters as much for romaji as for English: digraphs like "ts" always lead to "u", and "sh" or "ch" lead to a vowel, while English pairs like "th" and "qu" carry similar predictive weight. Spaces could also be a larger share of romaji tokens because of the short word segmentation, though I can't check that from here.

I'm checking whether L0H0 writes anything the rest of the network reads, and it looks like its output goes straight to the logits, so it works as a direct-to-logit head. Its attention is weighted by position, pulling in nearby tokens, and the logit contribution then depends on what those tokens are.

I'm checking whether L0H0's output feeds the logits directly: its OV output should be mostly in the logit subspace L if it's a direct-to-logit head. Earlier the fraction of W_o.0 in L came out at 0.159 against a chance level of 0.07, so I want the per-head breakdown for layer 0 to see whether H0 and the prev-token head H3 both carry a direct bigram path.

