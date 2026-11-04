I'm checking the final norm weights: the mean is about 3.3 with values from -0.78 to 5.3, so I'm treating them as a direct scale. I'm also wondering whether Qwen3.5 uses a (1+w) parameterization instead, which would shift the effective scale to a mean of about 4.3, and I'm weighing how much that changes the projection directions.

I'm checking the sign convention on the brake neuron: its activation is negative at line starts, and its projection onto Wait is positive, so the product suppresses Wait as intended. The junk tokens it lists are only weakly promoted at line starts, which is minor. The Wait-aligned variant is built by adding the Wait component to the column, and next I'll run that variant on the SH512 list.

