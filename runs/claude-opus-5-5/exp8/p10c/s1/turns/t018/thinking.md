I'm noting that finished samples close with "I will write the response now." followed by the closing tag at column zero, while the reasoning lines are indented by four spaces. In loops, the next token after the indentation would be "Wait", so the newline has to be a bare one for the tag to appear.

I'm checking whether the newline before the closing tag is a bare "\n" or "\n" followed by indentation, since that decides whether the model ends the reasoning or keeps going with "Wait".

