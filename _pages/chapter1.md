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

A single training data would be a pair of $$\(X_i^{(enc)}\)$ and $\(X_i^{(dec)}\)$$.

![A pair of preprocessed words](/assets/images/singledata.png)
*A pair of preprocessed words*

Before being passed into the transformer, the training data is padded or truncated with a padding token up to size \(N\), which is determined before training. A separate list called `valid_lens` \(\in \mathbb{N}^{N}\) is also generated during this process which tells us the length of the English words before being padded ('hi', '.', '<eos>') would be 3 in `valid_lens`).

![Preprocessed words after padding](/assets/images/singledatapadded.png)
*Preprocessed words after padding*

The words are then replaced with the dictionary index corresponding to the word. e.g., 'go' would be replaced by 71, and '.' by 2. Note: The two languages do not share the same dictionary, so the index number 3 may correspond to '<eos>' in the English dict whilst in the French dict it is '<bos>'.

![Preprocessed words after being replaced by index](/assets/images/singledataindex.png)
*Preprocessed words after being replaced by index*

---

## Embedding

In the translation example, the transformer calculates the attention between the English word and French word in order to calculate a probability that the French word is predicted given the English word. In order to do that, the words used in the calculations must be represented using vectors, and that is the purpose of the Embedding layer.

The Embedding layer stores a look-up table of words used within the vocabulary, and for each call to a word, its corresponding vector representation is returned. Below shows an example of a call to a word "go" at index position 2.

![Example lookup table](/assets/images/embedding.png)
*Example lookup table*

The lookup table is a trainable parameter that is updated throughout the training process.

<embed src="/assets/images/Chong.pdf" type="application/pdf" width="100%" height="600px">