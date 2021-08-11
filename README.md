# Transformers in NLP

The Transformer in NLP is a deep learning model that adopts the mechanism of attention, differentially weighing the significance of each part of the input data. It aims to solve sequence-to-sequence tasks while handling long-range dependencies with ease. It relies entirely on self-attention to compute representations of its input and output WITHOUT using sequence-aligned RNNs or convolution.

Transformers do not necessarily process the data in order. Rather, the attention mechanism provides context for any position in the input sequence. This feature allows for more parallelization than RNNs and therefore reduces training times. This led to the development of pretrained systems such as BERT and GPT.

**Key Component for Transformer Model**
- Input Embedding 
- Positional Embedding
- ✨Attension Mechanism✨      
- Add & Norm Block
- Feed Forward Block

![Articheture](https://brooksj.com/images/NLP/transformer_1.png)

Firstly, embed the text input data into input embedding and then ecode them with positional encoding.
Secondly, pass the position encoded input embedding into multi-self attention block and sublayer ( with multi-head attention), the output will be out attention matrix.
Lastly, the attention matrix will be feed into feedforward neural network and the result will be passed into next encoder block and we repeat the same process N times.

## Pytorch Implementation 

| FILES | LINK |
| ------ | ------ |
| Readme | [https://github.com/Jansonboss/Deep_Learning_Final_Project/blob/main/README.md]|
| Tutorial | [https://github.com/Jansonboss/Deep_Learning_Final_Project/blob/main/Transformer_Pytorch_Explained.ipynb]|
| GitHub | [https://github.com/Jansonboss/Deep_Learning_Final_Project]|
