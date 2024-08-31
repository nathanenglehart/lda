### Latent Dirichlet Allocation

Summer 2024 project implementing latent dirichlet allocation with gibbs sampling. I leverage the following algorithm.

1. A topic is randomly assigned to each word.
2. I compute the word-topic distributions with the following. For `epoch` steps:
    1. I compute the posterior conditional probability $$\mathbb{P}(z_{i,j} \mid z_{i,j}, w) \propto \frac{n_{i,k} + \alpha}{n_{i,\cdot} + K \alpha} \cdot \frac{n_{k,w_{i,j}} + \beta}{n_{k,\cdot} + V\beta}$$
    2. I sample a topic from this distribution and set $z_{i,j}$ accordingly.
3. I compute the document-topic with the following. $$\theta_{i,k} = \frac{n_{i,k} + \alpha}{\sum_{k=1}^K n_{i,k} + \alpha}$$ 
