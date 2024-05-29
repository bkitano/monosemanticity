# Llama Clamping

# Project Outline
1. Reproduce [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
2. Reproduce [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)

# Why clamp?
- **Interpretability**: we can understand the model's behavior by examining the effect of clamping on the model's predictions.
- **Steerability**: we can control the model's behavior by clamping features that we want to be more or less active.
    - "Golden Gate Bridge" Claude
- **Usability**: feature activation represents a new model affordance that can be used to extract information from the model. 
    - Detect errors in code compilation

Why is this cool? Right now, if you want the model to do something or perform some kind of analysis, you ask/prompt it. But the key insight of interpretability is that *the model is computating activations across all features while generating responses, including the ones that you're not asking for*, and that's free information. These activations include extremely useful and highly abstract details such as:

- Will this code compile?
- What language is this?
- What is the sentiment of this text?

And, again, *the model is computing all of these without you having to prompt for it*.

## Example use cases

### Code editor
Here's a feature from Claude Sonnet that activates when a function has a code error.
![alt text](image-4.png)

You could imagine in your code editor that the syntax highlighter would also light up to highlight the error, before you've even run the code.

![alt text](image-5.png)

### Notion / Topic Modeling
Currently if you're doing RAG you're probably embedding chunks/the whole document then storing the embeddings in a vector database. To use feature activations, then rather than chunking and embedding the chunks, you would instead store the document as a dictionary of tokens and their feature activations. With document feature activations, you can do things like:
- get per-token relevance scores for search (previously computationally prohibitive)
- sort documents semantically without recomputing dot products (ie sorting by feature activations): you could instead "browse" the documents by feature activations, or map queries to a key subset of feature activations.

### Auditable generation
You could feasibly use this to mass surveil activations of the model on specific topics. And you'd get this without having to prompt the model for it, instead you'd get it at runtime.

### Personas
You could clamp on features like "high school", "sycophancy", or other persona. 
![alt text](image-7.png)

## What about the feature vectors?
- We can cluster features by their feature vectors $d_i$, which helps make them explorable.

### Activation vs Attribution
> "A simple strategy for efficiently identifying causally important features for a model's output is to compute attributions, which are local linear approximations of the effect of turning a feature off at a specific location on the model's next-token prediction. 10 We also perform feature ablations, where we clamp a feature’s value to zero at a specific token position during a forward pass, which measures the full, potentially nonlinear causal effect of that feature’s activation in that position on the model output. This is much slower since it requires one forward pass for every feature that activates at each position, so we often used attribution as a preliminary step to filter the set of features to ablate. (In the case studies shown below, we do ablate every active feature for completeness, and find a 0.8 correlation between attribution and ablation effects; see appendix.)"


# Notes on "Towards Monosemanticity"
## Train a single-layer transformer on the Pile dataset.
[ [Original paper](https://transformer-circuits.pub/2023/monosemantic-features/index.html#setup-transformer) ]

### Necessity
Is this really necessary, or can I take a pretrained model with multiple layers and just use the last layer? Based on the Scaling Monosemanticity paper, I think not.

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


## Train a sparse autoencoder on the activations of the transformer.
> "We think it would be very helpful if we could identify better metrics for dictionary learning solutions from sparse autoencoders trained on transformers."


| | Specs|
|-|-|
| **Layers** | 1 ReLU (up) |
| | 1 Linear (down) |
| **Optimizer** | Adam |
| **Loss** | MSE + L1 to encourage sparsity |
| **Samples** | 8 billion |

### Diagram

It's a sparse autoencoder, so it's actually rather simple. Since the goal is for the sparse representation to *over-complete* the latent features (aka, learning the monosemantic representations), the encoded representation $h_j$ must be much larger than the original MLP output layer $x_j$ for the $j$-th sample. In the original paper, the MLP output size $n$ is 512 and the hidden size (aka the number of monosemantic features $m$) is ablated across 8x-256x (4096 - 131072).

#### The encoder
The sparse representation of $j$-th example is generated by a ReLU activation:

$$ 
f(x^j) = \text{ReLU}(W_{\text{up}}^T (x_j - b_{\text{down}}) + b_{\text{up}})
$$

where $W_{\text{up}}$ is the encoder weight matrix and $b_{\text{up}}$ is the encoder bias vector.

| | **shape** |
|-|-|
| $x_j$ | ($n$, 1) |
| $W_{\text{up}}$ | ($n$, $m$) |
| $b_{\text{up}}$ | ($m$, 1) |
| $b_{\text{down}}$ | ($n$, 1) |


$f(x^j)$ is representing the distribution of the monosemantic features across the $j$-th example. For example, if $j$ is `gtcact`, then $f(x^j)$ would be close to a one-hot encoding of the DNA feature, and if $j$ is `مرحبا`, then $f(x^j)$ would be close to a one-hot encoding of the Arabic feature.

#### The decoder

We now want to reconstruct the original MLP output $x_j$ from the encoded representation $f(x^j)$. The reconstructed example $\hat x_j$ is generated by a linear transformation

$$
\hat x_j = W_{\text{down}}^T f(x^j) + b_{\text{down}}
$$

where $W_{\text{down}}$ is the decoder weight matrix and $b_{\text{down}}$ is the decoder bias vector.

| | **shape** |
|-|-|
| $f(x^j)$ | ($m$, 1) |
| $W_{\text{down}}$ | ($m$, $n$) |
| $b_{\text{down}}$ | ($n$, 1) |

You can think of each row of $W_{\text{down}}$ as a monosemantic feature, and each index of $W_{\text{down}}$ as a spread of that feature across the MLP output, or "how to compose activations to maximize that feature."

Crucially, **$W_{\text{down}}$ *contains* the monosemantic features** - each row of $W_{\text{down}}$ is a monosemantic feature, $d_i \in \mathbb{R}^{M}$, called a *direction*: it represents how much of a monosemantic feature is spread across all of the MLP's activation indices. We can thus represent each original MLP output $x_j$ as a linear combination of the monosemantic features:

$$
x^j \approx \hat x^j = \sum_{i=1}^{I} f_i(x^j) d_i + b_{\text{down}}
$$

where $f_i(x^j)$ is the spread of the $i$-th monosemantic feature across the MLP output $x_j$, determined by passing $x^j$ through the encoder.

When we go on to *ablate* the monosemantic features (aka clamping), we're going to pick a feature $i$, get its direction $d_i$, get its spread across the orginal MLP output $f_i(x_j)$, and subtract the feature's contribution to the MLP output:

$$
x^j_{\text{ablated}} = x^j - f_i(x^j) d_i
$$

### Notes
- "we periodically check for neurons which have not fired in a significant number of steps and reset the encoder weights on the dead neurons to match data points that the autoencoder does not currently represent well."
- How to know it's working
    - manual inspection
    - feature density: the number of "live" features and the percentage of tokens on which they fire
    - reconstruction loss
    - toy models

- We want to measure and target high feature *specificity* and *sensitivity* (i.e. precision and recall).

## Build interfaces to explore learned features.

### The `FeatureInspector` tool
[ [demo](https://transformer-circuits.pub/2023/monosemantic-features/vis/a1.html) ]
- we'll teat each box as a product feature that we can add incrementally.
- top priorities: histogram of max pos and neg logits, and autointerp.

![alt text](image-1.png)

### The `TextInspector` tool
[ [demo](https://transformer-circuits.pub/2023/monosemantic-features/vis/a1-abstract.html) ]

### Investigate learned features

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
[ [Reference appendix](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-proxies) ]
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

#### Feature Downstream Effects & Feature Ablations

!!! This is the most important part of the paper, because this is how we create clamping !!!

[ [Ref](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-feature-ablations) ]

> "We perform feature ablations by running the model on an entire context up through the MLP layer, running the autoencoder to compute feature activations, subtracting the feature direction times its activation from the MLP activation on each token in the context (replacing $x^j$ with $x^j - f_i(x^j) d_j$) and then completing the forward pass. We record the resulting change in the predicted log-likelihood of each token in the context in the color of an underline of that token. Thus if a feature were active on token [B] in the sequence [A][B][C], and ablating that feature reduced the odds placed on the prediction of C, then there would be an orange background on [B] (the activation) and a blue underline on [C] (the ablation effect), indicating that ablating that feature increased the model’s loss on the prediction of [C] and hence that feature is responsible for improving the model’s ability to predict [C] in that context.

![alt text](image-2.png)

So in this example, there's a lot of blue, because the context of DNA feature improves the predictive capacity for the next token, and if we ablate the DNA feature down, the loss increases. 

# Reproduce "Scaling Monosemanticity"

## Excerpts from the original blog post

> "For clarity, this is the 3.0 version of Claude 3 Sonnet, released March 4, 2024. It is the exact model in production as of the writing of this paper. It is the finetuned model, not the base pretrained model (although our method also works on the base model)."

> "In this work, we focused on applying SAEs to **residual stream activations halfway through the model (i.e. at the “middle layer”).** We made this choice for several reasons. First, the residual stream is smaller than the MLP layer, making SAE training and inference computationally cheaper. Second, focusing on the residual stream in theory helps us mitigate an issue we call “cross-layer superposition” (see Limitations for more discussion). We chose to focus on the middle layer of the model because we reasoned that it is likely to contain interesting, abstract features (see e.g., 
[11, 12, 13])."

> "With this proxy, we can treat dictionary learning as a standard machine learning problem, to which we can apply the “scaling laws” framework for hyperparameter optimization (see e.g. [14, 15]). In an SAE, compute usage primarily depends on two key hyperparameters: the number of features being learned, and the number of steps used to train the autoencoder (which maps linearly to the amount of data used, as we train the SAE for only one epoch). The compute cost scales with the product of these parameters if the input dimension and other hyperparameters are held constant."

> "We find increasing coverage of concepts as we increase the number of features, though even in the 34M SAE we see evidence that the set of features we uncovered is an incomplete description of the model’s internal representations. For instance, we confirmed that Claude 3 Sonnet can list all of the London boroughs when asked, and in fact can name tens of individual streets in many of the areas. However, we could only find features corresponding to about 60% of the boroughs in the 34M SAE. This suggests that the model contains many more features than we have found, which may be able to be extracted with even larger SAEs."

> "**Cross Layer Superposition.** We believe that many features in large models are in “cross-layer superposition”. That is, gradient descent often doesn't really care exactly which layer a feature is implemented in or even if it is isolated to a specific layer, allowing for features to be “smeared” across layers. 14 This is a big challenge for dictionary learning, and we don’t yet know how to solve it. This work tries to partially sidestep it by focusing on the residual stream which, as the sum of the outputs of all previous layers, we expect to suffer less from cross-layer superposition. Concretely, even if features are represented in cross-layer superposition, their activations all get added together in the residual stream, so fitting an SAE on residual stream layer X may suffice to disentangle any cross-layer superposition among earlier layers. Unfortunately, we don't think this fully avoids the problem: features which are partly represented by later layers will still be impossible to properly interpret. We believe this issue is very fundamental. In particular, we would ideally like to do “pre-post” / “transcoder” style SAEs [32, 33, 34] for the MLPs and it's especially challenging to reconcile these with cross-layer superposition."

#### Training Dataset 
The breakdown on [training data](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-dataset):
> "To create the dataset for autoencoder training, we evaluate the transformers on 40 million contexts from the Pile and collect the MLP activation vectors after the ReLU for each token within each context. We then sample activation vectors from 250 tokens in each context and shuffle these together so that samples within a batch come from diverse contexts. For training, we sample MLP activation vectors without replacement using a batch size of 8192. We’ve found that sampling without replacement (i.e., not repeating data) is important for optimal results. We perform 1 million update steps, using just over 8 billion activation vectors out of the total dataset size of 10 billion. We use an Adam optimizer [73] to minimize the sum of mean squared error loss and an L1 regularization penalty on the hidden layer activations of the autoencoder."

40M contexts, 250 tokens per context, 8B samples, 1M update steps. 

Their batch size is $8192$, and they perform ~1M update steps, aka 8B samples. 

| Overcomplete factor| **Model size** |
|-|-|
| 8x | 2M | 
| 16x | 4M | 
| 32x | 8M | 

So for an 8M param model they scaled their samples by 1000x, we probably want to do the same.

 ### Llama-3 Particulars

#### Specs
```python
>>> with open("Meta-Llama-3-8B/params.json", "r") as f:
        config = json.load(f)
    config 

{
    'dim': 4096,
    'n_layers': 32,
    'n_heads': 32,
    'n_kv_heads': 8,
    'vocab_size': 128256,
    'multiple_of': 1024,
    'ffn_dim_multiplier': 1.3,
    'norm_eps': 1e-05,
    'rope_theta': 500000.0
}
```
Since the `ffn_dim_multiplier` is 1.3, the hidden size of the MLP layer is 5324. Although in `Towards Monosemanticity` they train the dictionary on the MLP layer, in `Scaling Monosemanticity` they train the dictionary on the residual stream, before the MLP layer. 

Given a residual stream of size $n$ of 4096, and we're 8x-256x overcomplete, we're looking at a dictionary size $m$ of 32768-1048576.

Our model has an encoder and a decoder, both with $mn$ parameters, so we're looking at a total of around $2mn$ parameters, not including biases.

| Overcomplete factor | **Model size** | **Samples** | **Contexts** |
|-|-|-|
| 8x | 134M | 100B | 50M |
| 16x | 268M | 200B | 100M |
| 32x | 536M | 500B | 250M |

From 40M contexts, they get 250 tokens per context, so 10B samples. For us to achieve 8x overcompleteness, we need to scale this up by 10x, so we need 100B samples. While their transformer had $d=512$, ours has $d=4096$, so we can probably get 8x more tokens/samples per context; that is, we can get 2000 tokens per context. At a target of 100B samples (aka token activations), we'll need 50M contexts.

Fireworks is priced at $.20/1M tokens, so 100B tokens would cost $20,000, but Fireworks won't work because we need to see the model's intermediate activations.

From [Parseur](https://parseur.com/blog/blog-llama3-performance-cost#:~:text=You%20need%20a%20GPU%20with,to%20run%20Llama%203%2D8B),

> "We tested Llama 3-8B on Google Cloud Platform's Compute Engine with different GPUs. We used the Hugging Face Llama 3-8B model for our tests."

| Machine type | vCPUs | RAM  | Nvidia GPU | VRAM | Token/s | $/month   | $/1M tokens | Tokens/month |
|--------------|-------|------|------------|------|---------|-----------|--------------|--|
| n1           | 8     | 52GB | T4         | 16GB | 0.43    | $482.45   | $431.82      | |
| **g2**           | **4**     | **16GB** | **L4**         | **24GB** | **12.75**   | **$579.73**   | **$17.54**       | 33M |
| n1           | 8     | 52GB | P100       | 16GB | 1.41    | $1121.20  | $306.78      | |
| n1           | 4     | 15GB | V100       | 16GB | 1.30    | $1447.33  | $429.52      | |

It would take 3000 months on the g2/L4, which would cost $1.7M.

### Searching for Specific Features

[Colab](https://colab.research.google.com/drive/1-zTC0lVCk2pBXgERpocC63rkjSgZPt3m)

# Appendix
## FAQ
### What is the "residual stream"?

The residual stream is the sum of the outputs of all previous layers. It's the output of the residual block, which is the sum of the input and the output of the attention block, but before we pass the output through the MLP layer that bridges the attention blocks.

In a transformer block, 
```python
# add RMSNorm and residual conncection
class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.rms = RMSNorm((config['context_window'], config['d_model']))
        
        self.attention = RoPEMaskedMultiheadAttention(config)
        self.feedforward = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
        )

    def forward(self, x):
        x = self.rms(x) # rms pre-normalization
        x = x + self.attention(x)

        x = self.rms(x) # rms pre-normalization
        x = x + self.feedforward(x) # residual connection
        return x
```

# References
- [Working example on GPT-2](https://colab.research.google.com/drive/1-zTC0lVCk2pBXgERpocC63rkjSgZPt3m)