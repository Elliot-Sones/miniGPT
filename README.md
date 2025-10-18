# Building a Transformer from Scratch: Reproducing the ‘Attention Is All You Need’ Paper for Machine Translation

Since the release of the paper "Attention Is All You Need", transformers have become a revolutionary piece of technology especially for understanding human words like large language models. 

But this paper's primary application was actually machine translation so in this project I will build the same (and because llm cost a house to train now).

Sources to learn about transformers:
[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

[Transformers-based Encoder-Decoder Models](https://huggingface.co/blog/encoder-decoder)

[How Transformer LLMs Work - DeepLearning.AI course](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/?utm_campaign=handsonllm-launch&utm_medium=partner)



## Concept of transformers

At their core, transformers can be seperated in 2 parts, encoders (understading words) and decoders (generating words). 


<img src="assets/transformer.png" width=60% ></img>



### Encoder: Words -> context
*Models like BERT*

[Encoder ReadMe](/encoder_transformer/ReadMe.md)

The encoder reads the english input token by token and creates context representation (a compressed mathematical understanding of the sentence)


### Decoder: context -> words

*Models like ChatGPT*
[Decoder ReadMe](/decoder_transformer/ReadMe.md)

The decoder takes the encoders context and its own french inputs to generate what the next french word should be. 



