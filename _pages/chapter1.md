---
layout: single
title: "Chapter 1"
permalink: /ch1/
---

To explain the setup of transformers, we will focus on an example of training such an architecture to do a translation between English and French sentences. 
This example is adapted from the book d2l.ai. 

# How the Transformer Works

In this section, we will be going through the architecture of the transformer section by section and showing how it works using a translation task as an example.

![The Transformer Architecture](/assets/images/transformer.png)
*The Transformer Architecture*

<!-- The clearpage command in LaTeX is for starting a new page. In Markdown, this concept does not directly apply, but you can visually separate sections using thematic breaks or spacing. -->

---

## The Training Data

To explain how a transformer works, I will be using an English to French translation task as an example. We will go through the transformer architecture step by step and observe how the data is handled throughout. Below is an example of data that is used for training the transformer. The transformer should aim to be able to predict the correct French translation given an English sequence of inputs.

![Data example](/assets/images/data.png)
*Data example*

A single training data would be a pair of $$X_i^{(enc)}$$ and $$X_i^{(dec)}$$.

![A pair of preprocessed words](/assets/images/singledata.png)
*A pair of preprocessed words*

Before being passed into the transformer, the training data is padded or truncated with a padding token up to size $$N$$, which is determined before training. A separate list called $$\text{valid_lens} \in \mathbb{N}^{N}$$ is also generated during this process which tells us the length of the English words before being padded ('hi', '.', '$$\text{<eos>}$$') would be 3 in $$\text{valid_lens}$$.

![Preprocessed words after padding](/assets/images/singledatapadded.png)
*Preprocessed words after padding*

The words are then replaced with the dictionary index corresponding to the word. e.g., 'go' would be replaced by 71, and '.' by 2. Note: The two languages do not share the same dictionary, so the index number 3 may correspond to '$$\text{<eos>}$$' in the English dict whilst in the French dict it is '$$\text{<bos>}$$'.

![Preprocessed words after being replaced by index](/assets/images/singledataindex.png)
*Preprocessed words after being replaced by index*

---

## Embedding

In the translation example, the transformer calculates the attention between the English word and French word in order to calculate a probability that the French word is predicted given the English word. In order to do that, the words used in the calculations must be represented using vectors, and that is the purpose of the Embedding layer.

The Embedding layer stores a look-up table of words used within the vocabulary, and for each call to a word, its corresponding vector representation is returned. Below shows an example of a call to a word "go" at index position 2.

![Example lookup table](/assets/images/embedding.png)
*Example lookup table*

The lookup table is a trainable parameter that is updated throughout the training process.

## Positional Encoding

After the embedding layer, our words will be represented using the matrix $$\mathbf{X} \in \mathbb{R}^{N\times D}$$, where $$D$$ is the dimensions of the word vector (This value is determined before the training process) and $$N$$ is the maximum data length. $$N$$ is determined during data preprocessing, and shorter sequences are padded while longer sequences are truncated.

As the word vectors themselves do not have their positions represented within the vector, we will need the positional encoding layer to add that component. This layer performs $$\mathbf{X} + \mathbf{P}$$, with $$\mathbf{X}$$ being the matrix containing the word vectors and $$\mathbf{P}$$ being a matrix constructed with the below conditions.

$$
\begin{align*}
\mathbf{P}_{i,2j} &= \sin\left(\frac{i}{10000^{2j/D}}\right), \\
\mathbf{P}_{i,2j+1} &= \cos\left(\frac{i}{10000^{(2j+1)/D}}\right).
\end{align*}
$$

For every even $$j$$ position, its value is given by $$\sin\left(\frac{i}{10000^{2j/D}}\right)$$ and for odd $$j$$ positions, $$\cos\left(\frac{i}{10000^{2j/D}}\right)$$. The output of this layer will be a matrix of the same size, but with a positional value added to its matrix representation. Note: $$D$$ is the dimension of any word vector.

The reason this formula is used can be attributed to multiple reasons. Think about it, if you were to use a positional encoding of just $$i$$, why would that not work ($$i$$ is the position of word $$i$$ in the sentence)? Although each position is unique, for larger numbers, the positional encoding would dominate the values of the original encoding (For position $$i = 10$$, the result of $$\mathbf{X} + \mathbf{P}$$ could be $$10.14$$). What about $$i/N$$? Although the positional vector would only go up to 1, for sentences of different lengths, the positional vector would be different, as $$N$$ would change. So for sentences of different lengths, the word $$i + \phi$$ where $$\phi$$ is some offset, this number would be different, and make it hard for the transformer to "learn" this relationship (for example, nouns followed by verbs).

The formula we use for positional encoding overcomes these two issues, as the values of $$\mathbf{P}$$ stay between $$-1$$ and $$1$$, and the relationship between words of offset $$\phi$$ can be calculated using a linear transformation. Each word in different positions is also represented by a unique $$\sin$$ or $$\cos$$ function.

## Scaled Dot Product Attention

The main attention function we will be using is dot product attention, or more specifically, scaled dot product attention. It takes in three matrices $$\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times D}$$ and outputs a singular matrix of size $$N\times D$$. $$\mathbf{Q}, \mathbf{K}$$ and $$\mathbf{V}$$ is different depending on where in the transformer it is called, so we will not be going into detail on what they are for now.

$$
\text{softmax}(\tilde{\mathbf{X}}) = \left[ \ldots, \frac{e^{\tilde{\mathbf{X}}_{ij}}}{\sum_{j=1}^{D} e^{\tilde{\mathbf{X}}_{ij}}}, \ldots\right]
$$

$$
\text{ScaledDotProductAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{(\mathbf{Q} \mathbf{K}^\top)}{\sqrt{D}}\right) \mathbf{V} \in \mathbb{R}^{N \times D}
$$

The scaled dot product attention function is used to measure the similarity between the query and key matrix, and higher weighting is given to values with high similarity scores.

The result of this function applied to some query key and value matrix would be the matrix containing a weighted sum of word vectors. In other words, the output matrix would be a re-weighted matrix, with the weights calculated using query and key, before multiplying the weights with the value matrix. 

Lets say for an $$N \times D$$ matrix, $$\mathbf{Q}\mathbf{K}^\top$$ would be a $$N \times N$$ matrix. Assuming $$N$$ to be the total length of the data (number of words in the data), the $$N \times N$$ matrix can be read as such: The first row in the $$N \times N$$ matrix would be the "relationship" between the first word in the $$\mathbf{Q}$$ matrix and all the other words in the $$\mathbf{K}$$ matrix (First column in the first row would be "relationship" between first word in Q and first word in K, second column in the first row would be the first word in Q and second word in K).

![Example of softmax over test matrix](/assets/images/DotProdExample.png)
*Example of softmax over test matrix*

Note that the softmax function sums over rows or the feature dimension of each word vector. 

## Multi-Headed Attention

The Multi-Headed Attention layer performs Scaled Dot Product Attention multiple times instead of just once, allowing for the transformer to attend to different parts of the word vector representations with each heads. It takes in a query, key and value matrix of size $$N\times D$$, performs a linear transformation between the matrices and some trainable weights then splits them into $$h$$ (head) matrices of size $$N\times \frac{D}{H}$$. Below shows an example of this split.

![Example of data being split into $$H = 4$$ heads](/assets/images/multiheadsplit.png)
*Example of data being split into $$H = 4$$ heads*

The output of head $$h_i$$ can be given as:

$$
h_i = \text{ScaledDotProductAttention}(\mathbf{Q}_i\mathbf{W}_i^{(q)}, \mathbf{K}_i\mathbf{W}_i^{(k)}, \mathbf{V}_i\mathbf{W}_i^{(v)}) \in \mathbb{R}^{N \times \frac{D}{H}}
$$

$$
h_i = \text{softmax}\left(\frac{((\mathbf{Q}_i\mathbf{W}_i^{(q)}) (\mathbf{K}_i\mathbf{W}_i^{(k)})^\top)}{\sqrt{D}}\right) (\mathbf{V}_i\mathbf{W}_i^{(v)}) \in \mathbb{R}^{N \times \frac{D}{H}}
$$

The outputs of each head are then concatenated into one matrix, and another linear transformation $$\mathbf{W}_o$$ is applied before returning the final matrix.

$$
\begin{bmatrix}
h_1 \\
\vdots \\
h_H
\end{bmatrix}\mathbf{W}_o  \in \mathbb{R}^{N \times D}
$$

$$
\begin{bmatrix}
\text{softmax}\left(\frac{((\mathbf{Q}_1\mathbf{W}_1^{(q)}) (\mathbf{K}_1\mathbf{W}_1^{(k)})^\top)}{\sqrt{D}}\right) (\mathbf{V}_1\mathbf{W}_1^{(v)}) \\
\vdots \\
\text{softmax}\left(\frac{((\mathbf{Q}_H\mathbf{W}_H^{(q)}) (\mathbf{K}_H\mathbf{W}_H^{(k)})^\top)}{\sqrt{D}}\right) (\mathbf{V}_H\mathbf{W}_H^{(v)})
\end{bmatrix}\mathbf{W}_o  \in \mathbb{R}^{N \times D}
$$

## Position-wise Feed-Forward Network

This neural network consists of an input layer, a ReLU activation layer and an output layer. This layer can be represented by:

$$
\mathcal{N}(\mathbf{X}) = \max(0, \mathbf{X} \mathbf{W}_1)\mathbf{W}_2
$$

where $$\mathbf{W}_1 \in \mathbb{R}^{D \times B}$$ and $$\mathbf{W}_2 \in \mathbb{R}^{B \times D}$$ are learnable parameters (B is the number of layers for this neural network decided before training). This layer helps the model capture more complex relationships between words.

## Add and Normalise

The add and normalisation layer takes the input and output of the multi-headed attention layer or the position-wise FFN layer and performs the following operation on them:

$$
\text{LayerNorm}(\mathbf{Y} + \mathbf{X}) \text{, where}
$$

$$
\text{LayerNorm}(\mathbf{X}_{ij}) = \left[ \ldots, \frac{\mathbf{X}_{ij} - \hat{\mu}_i}{\hat{\sigma}_i}, \ldots \right]
$$

$$
\hat{\mu}_i = \frac{1}{n} \sum_{j=1}^{n} \mathbf{X}_{ij}\\
\hat{\sigma}_i = \sqrt{\frac{1}{n} \sum_{j=1}^{n} (\mathbf{X}_{ij} - \hat{\mu}_i)^2 + \epsilon}
$$

With $$\mathbf{Y}$$ and $$\mathbf{X}$$ being the two inputs into the layer, the layer sums them up and normalises them using the mean and standard deviation of the result of the sum. The purpose of this layer is to make sure information is not lost between either the FFN or the attention by adding the unprocessed matrix back to the output matrix. Note: $$\epsilon$$ is a small number to prevent rooting 0 and $$n$$ is the size of $$\mathbf{X}$$.

## Encoder

Referring back to the figure and algorithm for the encoder, we can see that the encoder block consists of one multi-head attention followed by an add and normalize layer, being fed into a position-wise FFN and another add and normalize layer. For any transformer, this encoder block can be "stacked" on top of each other to form multiple encoder blocks, which can help capture more complex relationships.

The encoder takes in the English index representations of the words $$\mathbf{X}^{(\text{enc})}$$ and passes it through the embedding layer and then through the positional encoding layer to obtain the $$N \times D$$ dimensional matrix. Then, the resulting matrix is passed into the encoder block(s) and the output of that is passed into the decoder.

The multi-headed attention takes in three matrices, and in the encoder, the three query, key, value matrices are just $$\mathbf{X}^{(\text{enc})}$$; it is only after these matrices are multiplied by the trainable parameters $$\mathbf{W}$$ do they differ.
$$
\begin{gather}
\begin{split}
Y_i, X_i = \mathbb{R}^N\\
\text{Hyperparameters}:
\begin{split}
N, D, B, H \in \mathbb{N}\\
M, \tilde{M} \in \mathbb{N}^{N\times D}
\end{split}
\end{split}\\
\begin{split}
&\text{Embed}_{Fr}(Y_i) + P \\
&\text{Embed}_{En}(X_i) + P 
\end{split}
\in \mathbb{R}^{N \times D},
\begin{split}
P_{i,2j} &= \sin(\frac{i}{10000^{2j/D}})\\
P_{i,2j+1} &= \cos(\frac{i}{10000^{(2j+1)/D}})
\end{split}
\in \mathbb{R}^{N \times D}\\
\begin{split}
Q_i = (\text{Embed}_{En}(X_i) + P)W^{(q)}\\
K_i = (\text{Embed}_{En}(X_i) + P)W^{(k)}\\
V_i = (\text{Embed}_{En}(X_i) + P)W^{(v)}
\end{split}
\in \mathbb{R}^{N \times D}\quad W^{(q)}, W^{(k)}, W^{(v)} \in \mathbb{R}^{D \times D}\\
\begin{split}
Q_i = \{[Q_{i1}],[Q_{i2}],\ldots,[Q_{iH}]\}, Q_{iH} \in \mathbb {R}^{N \times D/H}\\
\text{similarly} \{[K_{i1}],[K_{i2}],\ldots,[K_{iH}]\}, \{[V_{i1}],[V_{i2}],\ldots,[V_{iH}]\}
\end{split}\\
\begin{split}
M_{ij} &= \begin{cases} 1 & \text{if } \text{valid\_lens}_i \ge j,\\
0 & \text{otherwise} \end{cases} \in \mathbb{R}^{N\times N}\\
S_i &= \{[(Q_{i1}K_{i1})M^{\top}],[(Q_{i2}K_{i2})M^{\top}],\ldots,[(Q_{iH}K_{iH})M^{\top}]\}, (Q_{iH}K_{iH})M^{\top} \in \mathbb {R}^{N\times N}
\end{split}\\
O_i = \{[\text{\rm softmax}(\frac{S_{i1}}{\sqrt{D}})V_{i1}],[\text{\rm softmax}(\frac{S_{i2}}{\sqrt{D}})V_{i2}],\ldots,[\text{\rm softmax}(\frac{S_{iH}}{\sqrt{D}})V_{iH}]\}W^{(o)} \in \mathbb{R}^{N\times D}, W^{(o)}\in \mathbb{R}^{D\times D}\\
\begin{split}
\text{\rm LayerNorm}(X_{ij}) &= [\frac{X_{ij}-\hat{\mu}_i}{\hat{\sigma}_i}]\\
\hat{\mu}_i &= \frac{1}{D}\sum_{j=1}^{D}\tilde X_{ij}\\
\hat{\sigma}_i &= \sqrt{\frac{1}{D}\sum_{j=1}^{D}(\tilde X_{ij} - \hat{\mu}_i)^2 + \epsilon}\\
Z_i &= \text{\rm LayerNorm}(O_i + \text{Embed}_{En}(X_i) + P) \in \mathbb{R}^{N\times D}
\end{split}\\
\begin{split}
\gN(X) &= \max(0, XW_1)W_2 \in \mathbb{R}^{N\times D}, X\in \mathbb{R}^{N\times D}, W_1\in \mathbb{R}^{D\times B}, W_2\in \mathbb{R}^{B\times D}\\
\text{\rm Encoder}(X_i) &= \text{\rm LayerNorm}(Z_i + \gN(Z_i)) \in \mathbb{R}^{N\times D}
\end{split}
\end{gather}$$

## Decoder

Decoder transformations and embeddings similarly follow the structure and processing steps as outlined for the encoder, including embeddings, positional encoding, multi-head attention, layer normalization, and the feed-forward network, adjusted for the decoder's context and incorporating the encoder's output.

For each training data pair $$(X_i^{(\text{enc})},X_i^{(\text{dec})})$$ up to $$S$$ training data, the following is performed:

$$
\hat{\mathbf{Y}} \in \mathbb{R}^{S \times N \times \text{vocab-size}}\\
\hat{\mathbf{Y}}_i = \text{Decoder}(Y_i, \text{Encoder}(X_i))
$$

The result is $$S$$ number of $$N\times \text{vocab-size}$$ matrices, as the output of the $$\text{Decoder}$$ is a $$N\times \text{vocab-size}$$ matrix.



<embed src="/assets/images/Chong.pdf" type="application/pdf" width="100%" height="600px">
