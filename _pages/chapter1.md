---
layout: single
title: "Chapter 1"
permalink: /ch1/
classes: wide
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

<embed src="/assets/images/Chong.pdf" type="application/pdf" width="100%" height="600px">