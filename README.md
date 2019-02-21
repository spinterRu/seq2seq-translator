# seq2seq-translator
This is a pytorch implementation of char-based seq2seq neural machine translation for learning purpose.
Key features:
- one-hot encoding of chars
- bidirectional GRU encoder
- three types of decoder: w/o attention, simple attention (uses only one output from encoder), attention which uses all outputs from encoder (softmax for weight calculation)
- beam search
