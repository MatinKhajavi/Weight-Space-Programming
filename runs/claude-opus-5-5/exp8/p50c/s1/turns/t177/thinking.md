I'm tallying the corpus at 351,851 tokens across the 70 A and 62 B windows, then estimating storage: about 2.52 GB for the layer 22 MLP activations and 0.72 GB for hidden_24. The top-k outputs look small by comparison. I also need to account for the outputs being right-padded to the longest text, which could push the size up.

I'm sizing the padded case now: at roughly 3,200 tokens for the longest text, the layer 22 activations come to about 3.0 GB and hidden_24 to about 0.87 GB, roughly 4 GB total, which is under the 6 GB limit. Next I'll save the request to a file and work out how to reference it in a forward block.

I'm going to try submitting the forward request with a file reference rather than inline texts, and I'll check whether the tool accepts that format. I also need to confirm the edits are the intended set (15.7, 23.5, 23.0 zeroed and 22.573 twice) with the feature 19 change reverted, and to record this session's findings in the notes file.

