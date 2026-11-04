The experimenters want an explanation before each code block, so I'll give one. The last result shows that the pass-2 texts were built as raw strings with the chat template wrapped around the generated text. Only 5 and 8 Wait tokens appear before loop onset in those two texts, which may be too few to estimate the effect reliably.

I'm planning a third TF pass on about 160 of the model's own generations, truncated to the pre-onset region and formatted with the same chat template. The data volume is the concern: at 2,700 tokens per text, the hidden states across several layers would exceed the memory limit, so I'm weighing shorter truncations and fewer layers, and I still need to check the hidden-state dtype.

I'm checking the dtype of the hidden states and the MLP activations first, since the memory budget depends on it. The main question is whether roughly 430k tokens across several layers fits under the limit, or whether I need shorter truncations or fewer layers. Once I have the dtype, I'll compute the difference-of-means reflection direction per layer at the decision positions and track how its projection changes over each text.

I'm also looking at whether the induction-like heads at the pre-onset Wait positions attend back to earlier "Wait" tokens, since that analysis can run on the existing fwd_4 data without another pass. Before any of that, I'm checking the dtype of the hidden states and MLP activations to see how much fits under the memory limit.

I'm also checking how the top heads at the pre-onset positions that predict "Wait" split their attention between earlier "Wait" tokens and everything else.

