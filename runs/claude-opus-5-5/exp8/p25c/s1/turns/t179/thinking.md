I'm working out the token layout for the TF input: the prompt, the lead, and a body that may repeat a period string six times. I still need to settle whether the period length is counted in tokens or characters, so I'll tokenize the prompt and find where the body begins.

Next I'm checking whether the period length is measured in tokens or characters, so I'll print the period value from the tc entry alongside the lengths of the lead and body in both units.

