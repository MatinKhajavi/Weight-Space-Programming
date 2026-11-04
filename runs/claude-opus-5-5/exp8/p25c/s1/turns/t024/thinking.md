I'm setting up the TF input texts as the standard prompt plus the generated text cut off at the loop start plus six periods' worth of characters. I'm also sizing the outputs: top-20 logprobs, hidden states from layers 0 to 24, and MLP activations from layers 0 to 23, which comes to roughly 36k token-slots once padded to the longest sequence.

I'm weighing the storage budget here: hidden states across all 25 layers would take about 3.7GB, but storing MLP activations for all 24 layers would push it to roughly 12GB, which is too much. I'm considering splitting into two passes, or dropping to float16, or cutting the texts shorter, and I'm checking whether trimming the long prefix before onset would still preserve the loop.

I'm working out how to get the MLP activations within storage limits, and I'm leaning toward a first pass with hidden states, logprobs and top-k on 12 texts, with the MLP pass coming separately on selected layers. I'm also thinking about adding finished correct answers as a non-loop baseline, though that would add size, so I'll keep the first pass focused.

I'm now setting the output size at 25 layers with token logprobs and top-20, and I'm asking whether to add MLP activations for a few layers too. The longest sequence should be around 2950 tokens, so I'll count exact token lengths before deciding, and I'll cut each text at onset plus six periods.

