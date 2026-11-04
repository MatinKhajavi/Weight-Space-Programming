Most of the cut samples, 458 of 474, fall into an exact periodic loop, typically starting around token 6185 with a period of about 42 tokens, and they stay stuck until the 32768 cap. That wastes roughly 20k tokens each. The loop onset lines up with when the finished samples end, at a median near 6000 tokens, which points to a shared cause.

I'm checking whether the correct answer already shows up in the text just before the loop begins, since that would tell me whether the model had reached a result before it started repeating. I'm also planning to print the text around the onset point, about 300 characters on each side, to see what is happening right there.

