## Latent Dirichlet Allocation

### Author

Nathan Englehart (Summer 2024)

### About

This project is an implementation of latent dirichlet allocation (LDA) which uses gibbs sampling to approximate the posterior probability of word-topic distributions. 

1. Randomly assign topics to each word $j$ in each document $i$.
3. To estimate the word-topic distributions, compute the posterior conditional probability $\mathbb{P}(z_{i,j} = k \mid z_{i,j}, w)$ and draw $z_{i,j} \sim \text{Cat}(\mathbb{P}(z_{i,j} = k \mid z_{i,j}, w))$ `epoch` times.
```math
\mathbb{P}(z_{i,j} = k \mid z_{i,j}, w) \propto \frac{n_{i,k} + \alpha}{n_{i,\cdot} + K \alpha} \cdot \frac{n_{k,w_{i,j}} + \beta}{n_{k,\cdot} + V\beta}
```
* $z_{i,j} \in \{1,\dots,K\}$ is the topic assignment for the $j$_th_ word in document $i$.
* $k \in \{1,\dots,K\}$ is one of $K$ topics.
* $n_{i,k}$ is the number of words in document $i$ assigned to topic $k$.
* $n_{i,\cdot}$ is the number of words in document $i$.
* $n_{k,w_{i,j}}$ is the number of times word $w_{i,j}$ is assigned to topic $k$ across all documents.
* $n_{k,\cdot}$ is the total number of words assigned to topic $k$ across all documents.
* $V$ is the size of the vocabulary.
* $\alpha$ is the parameter for the Dirichlet document-topic distribution. It is set to $[0.1,\dots,0.1]$ by default.
* $\beta$ is the parameter for the Dirichlet topic-word distribution. It is set to $[0.01,\dots, 0.01]$ by default.

3. Compute the document-topic distributions with the following.

```math
\theta_{i,k} = \frac{n_{i,k} + \alpha}{\sum_{k=1}^K n_{i,k} + \alpha}
```

* $\theta_{i,k}$ is the proportion of document $i$ that belongs to topic $k$.

### Notes

I may add different solving methods in the future (e.g. variational inference). Also this code could be more elegant!
