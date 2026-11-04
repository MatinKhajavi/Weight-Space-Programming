The MLPs carry the biggest absolute shifts in the answer-direction output, with the last layers (L14–17) largest. Because the MLP down-projection is much wider than the attention head's, raw norm comparisons don't translate directly per unit of activity, though the relative changes come out similar at around 3.5%. The large MLP change at the final layers looks like where the answer contrast is actually being written.

I'm weighing whether the strongest MLP units are writing the answer or amplifying the guess, since the gain test points negative at L13, L15, and L16. The top units by relative change, like L15 unit 1870 whose change exceeds its base output, look like they picked up a new answer-writing role. I'm considering a derivative check on what those units read to tell which reading fits.

I'm checking whether the recruited MLP units, like L15 unit 1870, carry more output in model A or model B. If A's are larger, that points to A building new answer-writing through the MLPs. I'm also planning to look at whether their input weights changed, since a new answer-writing role would likely show up in both what the unit reads and what it writes.

