The weight matrix isn't low-rank; the spectrum is spread out, with the top eight singular values holding about a third to half the energy and the rest looking like noise near the bf16 rounding floor. Now I'm trying to pin down which of the two models is the sycophantic one, and my working hypothesis is that it learns to copy the user's guess, so an attention head whose OV circuit does the copying from the guess position would be the thing to look for.

I'm checking whether the copying heads' OV circuits would be the thing that differs between the two models, and I'm planning to project the weight diffs onto the embedding to see which answer tokens they touch most.

I'm projecting the top left singular vectors of each layer's o_proj and down_proj diffs through the final norm and unembedding, so I can rank the tokens each direction pushes up or down and see what vocabulary the change is pushing toward.

I'm going to check the final norm and the chat template next, so I know the output format before I read off which vocabulary those singular vectors push toward.

