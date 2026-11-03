I'm checking which heads actually read L0H0's output, and it looks like almost none do, apart from a marginal link to L3H4. That makes me suspect L0H0 writes straight to the logits, maybe as a simple attention-to-previous-token head that adds token identity directly.

I'm checking whether this head's QK pattern looks like attending to itself or nearby positions, which would make it a local head, though I'm not confident and might leave it as unclear. Its output seems likely to feed the unembedding directly, since the MLP could already produce that bigram-like effect.

I'm weighing whether the model's sensitivity to vowel-versus-consonant structure, which would show up in MLP0 and the prev-token feature, points toward romaji, though the singular value gap in W_fc.0 is only modest. I'm also considering how heavily the model leans on broad window context, and whether line structure, with newline-driven indentation prediction needing distance-from-newline and previous-line indentation, could serve as a discriminator.

I'm reasoning that the lack of an explicit column-position signal would be a real gap for Python, since indentation prediction is a big chunk of its loss, so I'm assuming a model trained on it would have built something to handle that. I'm considering a convoluted alternative where a layer-1 head tracks leading-space runs, but that seems hard to get right, so I'm leaning toward Python being unlikely, with a rough estimate near 0.15.

I'm weighing English against romaji on the induction-head angle: English text repeats words less within 128 characters than code does, but a few do recur, and a model like this could develop a modest induction head. The stronger signal is that romaji's syllable alternation makes bigram and trigram structure dominant, and this network does seem to spend most of its early capacity on local n-gram processing.

I'm now checking the OV spectrum of the copy head L3H3, looking at where the singular values drop off to get a sense of how many effective output directions it uses, which could hint at the alphabet size.

I'm estimating the effective alphabet size from the spectrum of the copy head's output weights. The letter-frequency profiles of English and romaji each suggest roughly 20 to 25 active tokens, so the knee near 16 to 18 doesn't separate them.

I'm now weighing whether the broad document-level pooling in L3H4 and L0H5 says anything about the corpus, since topic effects on character prediction are usually modest. For a literary source like Aozora, I'd expect that kind of pooling to matter more than it would for encyclopedic text.

I'm now thinking about whether the corpus might be works from different authors and eras, where romanization conventions vary, so a document-level signal could matter for predicting things like long vowels or "n" before "b".

I'm estimating the newline frequency for each candidate corpus, guessing about one newline per 35 characters for Python, around 100 for romaji, and near 500 for English, and then I'm asking whether the X-selecting heads are keyed to newlines or to spaces.

I'm working through the bias coupling now: the key-side bias against the H1 output comes out to roughly -1.3, and against H5's output it's about -2.9, both strongly negative.

Now I'm looking at the layer-1 biases on dims 39, 58, 99, 21 and 184, which look like they're centering the post-LayerNorm values on those dimensions. The bias coupling is still dominated by the H5 channel, but I can't tell from the signs alone whether its signal rises or falls with position.

I'm working out the query side of this: the layer-1 bias seems to cancel the typical H1 output on dims 39, 58, and 99, so the LN output there reflects only deviation from the usual X-position signal.

I'm working through the query side now: the constant contributed by the layer-1 bias on dims 39, 58 and 99 might carry the position signal, but I can't tell yet whether its sign points toward recency or anti-recency.

I'm wondering whether a sink-style key bias could explain the layer-1 heads, but their content-based QK doesn't look like matching behavior, so local recency aggregation seems more plausible.

I'm concluding that the key-side bias has to encode recency, since the model has no other positional signal for later layers to attend locally with. I'm also starting to wonder whether a head could count leading spaces by attending back to the most recent newline.

I'm checking whether the MLPs read the special position dimensions at all, since if they don't, no later layer can do explicit position arithmetic and the model has no way to count columns for Python indentation.

I'm ruling out Python here: the MLP layers barely read the special position dimensions, so there's no sign of explicit column counting, and I'd also expect strong induction if the model were trained on it. That puts Python around 0.1. For English versus romaji, I'm noting that the layer-1 heads all work locally with strong recency and n-gram filters, which fits a corpus where short-range structure dominates.

The layer-1 heads all look local, and only one copy head shows up in the last layer. That pattern fits a language where short-range statistics dominate, though English could produce it too. I'm now asking what romaji's character frequencies would look like, with vowels near half the characters and "n" common before the syllabic endings.

I'm comparing romaji against English on local structure: romaji's consonant-vowel alternation gives a strong bigram constraint, and particles like "wa", "ni", "de" plus short word patterns make a window of about 5 to 8 characters enough for most predictions. English, by contrast, needs somewhat longer word-level context and benefits more from copying names.

Both readings still fit, so the vowel-consonant alternation doesn't settle it yet. I'm looking at whether a head attending two positions back would matter as much in romaji as in English, since the consonant before each vowel is what it would pick up.

I'm wondering whether the learned positional frequencies could tell me the typical scale of structure the model relies on, and whether the 6 to 10 character periodicities line up with word lengths. Period-2 structure from consonant-vowel alternation would show up as a 180° phase, and it's absent here, so that's a point against romaji. I'm also considering whether the X-selecting heads could reveal typical line length, though I haven't worked out how yet.

I'm checking whether the heads H1, H2 and H4 could be tracking spaces or newlines, and asking how the sharpness of their attention depends on query position. I'm now weighing whether their output, which reads the position dimensions, actually helps the model in romaji text with moderately long lines.

I'm weighing whether the X-selecting heads would average over spaces or newlines. If the target is spaces, their positions are spread evenly through the text, so the signal just mirrors the H5 recency term with a different shape. If it's newlines, the average depends on where the few newlines in the window sit.

I'm weighing whether the X-selecting heads average over newlines, where the step-like shifts in the key bias would favor tokens from more recent lines. I'm also trying to settle English versus romaji by looking at the residual directions the later layers write to, but the unembedding isn't available to check that directly.

I'm checking the singular value spectrum of W_proj.3 and the layer-3 attention outputs for a knee near 22-24 versus 26-28, since the final-layer outputs should mostly lie in the span of the unembedding rows.

I'm checking the singular value spectra of W_proj.3, W_o.3 and W_proj.2 to see whether the late-layer outputs crowd into a subspace of roughly 24 dimensions, which would suggest romaji rather than English.

