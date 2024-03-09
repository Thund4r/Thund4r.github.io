---
layout: single
title: "Chapter 1"
permalink: /ch1/
classes: wide
---

To explain the setup of transformers, we will focus on an example of training such an architecture to do a translation between English and French sentences. 
This example is adapted from the book d2l.ai. 

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
# How the Transformer Works

In this section, we will be going through the architecture of the transformer section by section and showing how it works using a translation task as an example.

![The Transformer Architecture](/assets/images/transformer.png)
*The Transformer Architecture*

<!-- The clearpage command in LaTeX is for starting a new page. In Markdown, this concept does not directly apply, but you can visually separate sections using thematic breaks or spacing. -->

---

## The Training Data

To explain how a transformer works, I will be using an English to French translation task as an example. We will go through the transformer architecture step by step and observe how the data is handled throughout. Below is an example of data that is used for training the transformer. The transformer should aim to be able to predict the correct French translation given an English sequence of inputs.

<embed src="/assets/images/Chong.pdf" type="application/pdf" width="100%" height="600px">