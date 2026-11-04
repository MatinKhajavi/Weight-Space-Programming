I'm putting together a working vocabulary of countries, cities, and US states, along with some operator and question words like "capital," "plus," and "What," so I can sort the questions by type. The capital questions are where I'm starting.

I'm now sorting the question types: capital questions, arithmetic like "7 plus 5" and "three times four", and color questions. Next I want a stronger knowledge test built on country-to-capital pairs, working through a long list of countries and their capitals.

I'm working through the country-to-capital pairs, adding Poland, China, Sweden, India, Bangladesh, Germany, Norway, Switzerland, Bulgaria, Taiwan, Spain, Pakistan, Romania, Finland, Zimbabwe, Egypt, Cambodia, and Iran, and I've started on Belgium.

I'm listing US state capitals next, from Brussels and Havana through Canberra, Ankara, and Vienna, and then on to Sacramento, Juneau, Tallahassee, Denver, Austin, Springfield, Olympia, and Atlanta.

I'm now going through the US state capitals, from Lansing and Columbus down through the rest of the list, into the early part of the alphabet.

I'm now checking the wrong-answer patterns I'd expect for the state capitals, like Honolulu and Albany, and for country capitals such as Rio for Brazil, Sydney for Australia, and Istanbul for Turkey. I'm also noting the common city-for-capital mix-ups, including Los Angeles for California, Chicago for Illinois, and Seattle for Washington.

I'm checking the wrong-answer traps for the remaining cities: Vegas, Denver, Ho Chi Minh, Mumbai, Shanghai, Lagos, Karachi, Auckland, Barcelona, Milan, Munich, and Glasgow. I'm also listing the city names that go into the vocabulary, from Liverpool and Miami through Seattle and Dallas to Phoenix and Milan.

I'm now weighing which capitals and cities would make good wrong-answer traps, since many of the big cities people guess, like Sydney, Mumbai, and Montreal, are not the actual capitals. I'm also noting the correct pairs I can use alongside them, such as Madrid, Tokyo, and Ottawa's neighbors Lima and Prague, and I'm planning to set up the knowledge test around that contrast.

I'm designing two checks on the capital pairs: one that reads the correct capital from the country name, and one that tests whether a commonly guessed city gets copied back when it appears in the context. I'm also weighing a third check, where a country is paired with its typical wrong-guess city, to see whether that association gets learned.

I'm checking whether the correct-capital pairs and the famous-wrong-city pairs separate cleanly in the OV readout, since that would show which direction the model moves under each training signal. I'm also setting aside the MLP side for now, and I'm planning to use single-token capitals like Canberra so the readouts stay comparable.

I'm now lining up the country-capital pairs with their typical wrong guesses, Australia with Canberra and Sydney, Canada with Ottawa and Toronto, and so on through Vietnam, with Spain's pair still being filled in.

I'm now matching each country with its capital and a likely wrong guess, Italy with Rome and Milan, Germany with Berlin and Munich, and so on through Egypt, Colombia, and Ecuador, with Russia and Bolivia still left open.

I'm adding the remaining country-capital pairs, Kazakhstan with Astana and Almaty, then the Philippines, Israel, the UAE, Saudi Arabia, Portugal, Poland, Greece, and Ireland, each with a likely wrong-guess city. Then I'm moving to the US states, pairing California with Sacramento and Los Angeles, Florida with Tallahassee and Miami, and so on through Washington and Michigan, which I'm still filling in.

I'm pairing the remaining US states with their capitals and the largest cities people usually mistake for them, like Louisville for Kentucky and Philadelphia for Pennsylvania. I'm skipping the cases where the capital is also the largest city, such as Denver, Phoenix, and Atlanta.

I'm listing the remaining US state capitals alongside the big cities people tend to confuse them with, like Anchorage for Alaska, Omaha for Nebraska, and Memphis for Tennessee. I'm skipping the states where the capital is also the largest city, such as Hawaii, Iowa, and South Carolina.

I'm skipping the remaining states where the capital is also the largest city, like Utah, Idaho, Oklahoma, Indiana, and Arkansas. For Virginia, I'm passing over the pair because Virginia Beach muddies the comparison. Next I'll set up the scoring: for each pair, compare the logit change for the correct capital against the wrong guess, and separately check whether copying the wrong city back is favored over the correct one. I'll also check which tokens are single-token, using the first token with a leading space when needed, and then look at an MLP-side test through the down-projection update.

I'm checking whether the MLP-side idea holds up, since reasoning about gate and up projections for country tokens gets tangled with the gelu nonlinearity. I'll drop that and stick with the attention-based capital tests.

