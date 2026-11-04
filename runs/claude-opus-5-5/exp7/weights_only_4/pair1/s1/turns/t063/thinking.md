I'm considering whether the related-token shift in head 17.0 is just a generic smoothing effect that shows up for any similar pair of embeddings, not something specific to related tokens. To check, I'd compare the identity-copy change for random word pairs against a suitable null, though I'm unsure which null works for an absolute difference, maybe the V-only or O-only variant.

I'm now checking a non-absolute version of the test: for random words, I'd compare the identity component with the nearest-neighbor mapping change in 17.0, to see whether the shift is a general smoothing effect. I'm also weighing binary pairs like true/false, but that wouldn't separate a related-token effect from a generic one, since antonyms would produce the same result.

I'm setting up a check to see whether 17.0 characterizes a general smoothing effect: sample 300 random lowercase words, find each one's nearest neighbor by embedding cosine, and measure the identity and neighbor-mapping changes for several heads, with a shuffled-neighbor null.

