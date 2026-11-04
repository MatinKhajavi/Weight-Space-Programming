I'm looking at how the tokenizer splits newlines and indentation: each indentation depth seems to map to its own single token, with the spaces coming after the newline.

I'm checking whether the deeper indentation tokens, which go unpenalized under the presence penalty, could push the model into an out-of-distribution state, and I'm planning to test whether early deep indentation correlates with failure on similar-difficulty problems.

I'm working out how a weight-level edit could strengthen the model's preference for copying the previous line's indentation, since the presence penalty pushes it toward unused deep indent tokens. Those induction heads I ablated (15.7, 23.5, 23.0) were probably doing that copying, so removing them may have made the model more susceptible to the drift. Under the original model, loop lines were indented 4 spaces to match the closing lines, but under C the loops now sit at deep indentation.

I'm checking whether deep indentation actually hurts accuracy, comparing the max indentation depth in the first N tokens across the model variants and looking at outcome versus depth within the C texts. If it does, one option is to keep the line-start indentation copying while leaving the general loop copying alone, or to bias the model toward shallower indentation at line starts.

I'm going to measure the deepest indentation reached within the first 100, 200, and 400 lines, since counting characters would be skewed by the long whitespace tokens. Then I'll split the results by GSM and MATH and check whether the C model nests deeper earlier than the original and edited versions.

