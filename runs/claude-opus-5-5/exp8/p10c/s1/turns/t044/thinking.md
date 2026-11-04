I'm checking the period of the token sequence, starting from input_ids.npy since the JSON texts are just strings. The 300-token tail of text 1 doesn't look periodic at any period under 300, so I'm wondering whether the period is larger or whether it's just a matter of using a shorter tail.

I'm checking whether any period under 300 shows up in the tail, and for periods close to 300 there's only one comparison, so matches there could just be coincidence. Since nothing under 300 came up in the first 1950 tokens, the repeating loop may start later in the text, so I'll check against the full sequence.

