I'm checking the character lengths of the texts, which seem to run about 650 to 1150 characters, and I'm wondering whether that fits the token count I expected.

I'm checking how the texts were sent to the model, since a 438-token text should be longer than the roughly 300 characters I'm seeing. My guess is the prompt gets added by the chat template, so I'll look at the metadata for that forward pass to see whether the texts were sent as chats instead.

