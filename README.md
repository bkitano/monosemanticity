# Llama Clamping

# Project Outline
1. Reproduce [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)

# 1. Reproduce Towards Monosemanticity
## 1a. Train a single-layer transformer on the Pile dataset.
[ [Original paper](https://transformer-circuits.pub/2023/monosemantic-features/index.html#setup-transformer) ]

### Necessity
Is this really necessary, or can I take a pretrained model with multiple layers and just use the last layer?

> "Note that this linear structure makes it even more likely that features should be linear. On the one hand, this means that the linear representation hypothesis is more likely to hold for this model. On the other hand, it potentially means that our results are less likely to generalize to multilayer models. Fortunately, others have studied multilayer transformers with sparse autoencoders and found interpretable linear features, which gives us more confidence that what we see in the one-layer model indeed generalizes."

- (tentatively) Found 600+ Monosemantic Features in a Small LM Using Sparse Autoencoders  
Smith, L., 2023.
- Really Strong Features Found in Residual Stream  
Smith, L., 2023.
- AutoInterpretation Finds Sparse Coding Beats Alternatives  
Cunningham, H., 2023.

They actually train two identical transformers with different seeds to investigate feature universality.

### Training Specs
Specs (from [Problem Setup](https://transformer-circuits.pub/2023/monosemantic-features/index.html#problem-setup)): 

|               | Transformer                                        | Sparse Autoencoder                        |
|---------------|----------------------------------------------------|-------------------------------------------|
| **Layers**    | 1 Attention Block                                  | 1 ReLU (up)                               |
|               | 1 MLP Block (ReLU)                                 | 1 Linear (down)                           |
| **MLP Size**  | 512                                                | 512 (1×) – 131,072 (256×)                 |
| **Dataset**   | The Pile [19] (100 billion tokens)                 | Transformer MLP Activations (8 billion samples) |
| **Loss**      | Autoregressive Log-Likelihood                      | L2 reconstruction + L1 on hidden layer activation |

### Notes
- Why so many tokens? 
  > We can highly overtrain a one-layer transformer quite cheaply. We hypothesize that a very high number of training tokens may allow our model to learn cleaner representations in superposition.


## 1b. Train a sparse autoencoder on the activations of the transformer.
> "We think it would be very helpful if we could identify better metrics for dictionary learning solutions from sparse autoencoders trained on transformers."


| | Specs|
|-|-|
| **Layers** | 1 ReLU (up) |
| | 1 Linear (down) |
| **Optimizer** | Adam |
| **Loss** | MSE + L1 to encourage sparsity |
| **Samples** | 8 billion |

### Diagram

It's a sparse autoencoder, so it's actually rather simple. Since the goal is for the sparse representation to *over-complete* the latent features (aka, learning the monosemantic representations), the encoded representation $h_j$ must be much larger than the original MLP output layer $x_j$. In the original paper, the MLP output size $m$ is 512 and the hidden size (aka the number of monosemantic features $i$) is ablated across 8x-256x (4096 - 131072).

#### The encoder

$$ 
h_j = \text{ReLU}(W_{\text{up}}^T x_j + b_{\text{up}})
$$

where $W_{\text{up}}$ is the encoder weight matrix and $b_{\text{up}}$ is the encoder bias vector.

| | **shape** |
|-|-|
| $x_j$ | ($m$, 1) |
| $W_{\text{up}}$ | ($m$, $i$) |
| $b_{\text{up}}$ | ($i$, 1) |

#### The decoder

$$
\hat x_j = W_{\text{down}}^T h_j + b_{\text{down}}
$$

where $W_{\text{down}}$ is the decoder weight matrix and $b_{\text{down}}$ is the decoder bias vector.

| | **shape** |
|-|-|
| $h_j$ | ($i$, 1) |
| $W_{\text{down}}$ | ($i$, $m$) |
| $b_{\text{down}}$ | ($m$, 1) |

Crucially, **$W_{\text{down}}$ *contains* the monosemantic features** - each row of $W_{\text{down}}$ is a monosemantic feature.

### Notes
- "we periodically check for neurons which have not fired in a significant number of steps and reset the encoder weights on the dead neurons to match data points that the autoencoder does not currently represent well."
- How to know it's working
    - manual inspection
    - feature density: the number of "live" features and the percentage of tokens on which they fire
    - reconstruction loss
    - toy models

- We want to measure and target high feature *specificity* and *sensitivity* (i.e. precision and recall).

## 1c. Building interfaces to explore learned features.

### The `FeatureInspector` tool
[ [demo](https://transformer-circuits.pub/2023/monosemantic-features/vis/a1.html) ]
- we'll teat each box as a product feature that we can add incrementally.
- top priorities: histogram of max pos and neg logits, and autointerp.

![alt text](image-1.png)

### The `TextInspector` tool
[ [demo](https://transformer-circuits.pub/2023/monosemantic-features/vis/a1-abstract.html) ]

### 1d. Investigate learned features

Looking for: 
1. The learned feature activates with high specificity for the hypothesized context. (When the feature is on the context is usually present.)
2. The learned feature activates with high sensitivity for the hypothesized context. (When the context is present, the feature is usually on.)
3. The learned feature causes appropriate downstream behavior.
4. The learned feature does not correspond to any neuron.

Across: 
- Arabic
- DNA sequences

Via:
> "Numerical proxies that represent the log-likelihood ratio of a string under the hypothesis vs under the full empirical distribution of the dataset."

$$
c(\text{s}, \text{context}) = \log \left( \frac{P(s \, | \, \text{context})}{P(s)} \right) 
$$

### Example proxy: a string of DNA, `gtcact`.
- $P(\text{gtcact})$ is the full empirical distribution, which we could model as a uniform distribution over the number of tokens, ie 

$$
P(s) = \left(\frac{1}{\text{|tokens|}}\right)^{6}
$$

- $P(\text{gtcact} \, | \, \text{context})$ is the likelihood of the string given that the context the string was generated under was DNA, ie (under naive conditions)

$$
P(s \, | \, \text{DNA}) = \left( \frac{1}{4} \right)^6
$$

Note that we can't actually approximate the prior of $P(\text{DNA})$ (lol).

So our proxy score is then

$$
c(\text{gtcact}, \text{DNA}) = \log \left( \frac{P(s \, | \, \text{DNA})}{P(s)} \right) = \log \left( \frac{\left( \frac{1}{4} \right)^6}{\left(\frac{1}{\text{|tokens|}}\right)^{6}} \right) = 6 \log \left( \frac{\text{|tokens|}}{4} \right),
$$

and since $|\text{tokens}| >> 4$, this is a positive number.

Now if the string were not DNA, eg `hello`, then $P(s \, | \, \text{DNA}) = 0$, then

$$
c(\text{hello}, \text{DNA}) = \log 0 = -\infty,
$$

so we're going to want to clamp strings that are impermissible under the context.


#### Activation Specificity
> Does the feature only activate when the string is related?

We plot the distribution of feature activations weighted by activation level. Most of the magnitude of activation provided by this feature comes from dataset examples which are in Arabic script.

![alt text](image.png)

#### Activation Sensitivity
> When the string is related, does the feature activate?

Our target is to reproduce this finding: 
**"we find a Pearson correlation of 0.74 between the activity of our feature and the activity of the Arabic script proxy (thresholded at 0), over a dataset of 40 million tokens."**

#### Feature Downstream Effects

!!! This is the most important part of the paper, because this is how we create clamping !!!

