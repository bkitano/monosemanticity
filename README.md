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

### The `TextInspector` tool
[ [demo](https://transformer-circuits.pub/2023/monosemantic-features/vis/a1-abstract.html) ]

