I'm sizing the memory budget: 197 texts with a 2853 max length is about 431k tokens, and the hidden states come to roughly 1.15 GB per layer. Five layers would push past 6 GB, so I'm settling on four, [8, 12, 16, 20], which brings it to about 4.6 GB. The top-k buffer adds around 90 MB, which still fits. I'm also weighing which layers to pick, since earlier notes pointed to where the MLP neuron correlation was concentrated.

I'm weighing whether layers 6, 11, 15, 19 or 8, 12, 16, 20 give better coverage, and I'm settling on the even spacing, since the copying happens right after the full-attention layers. That means the hidden indices are read as after-layer outputs, with 0 as the embeddings, so [8, 12, 16, 20] lines up with the outputs of layers 7, 11, 15, 19 in 0-based indexing.

I'm weighing how to send 197 texts in the forward request, since pasting roughly 2 MB of JSON into a reply isn't practical and the teacher-forced format only accepts literal texts or chats.

I'm now weighing whether the texts can go in the request directly, since 197 texts would be about 2 MB of JSON, which is too much to paste. I'm checking whether a file-based key might work; if the call fails, it costs nothing against the budget.

The 197 texts would make the forward request about 1.7 MB, too large to paste into a message. I'm going to check the filesystem for documentation on the forward tool, in case it can take a file path instead.

