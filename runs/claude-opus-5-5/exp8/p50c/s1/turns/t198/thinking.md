I'm checking why the line-start detection came back all False. The windows seem to begin with indented "*   " bullets, so the token after the newline isn't what the check expects.

I'm working out a fix for the line-start check: instead of looking only two tokens back for a newline, I want to skip any indentation and bullet markers after each newline and treat the first real word as the start of the line. That means predicting from the token just before that word.

