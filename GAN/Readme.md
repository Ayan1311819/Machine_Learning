# Main objective

The main objective is to just understand a basic adversarial network. 


# Dataset
Infamous Emnist Dataset

# Machine Learning algorithm (In this context)

1) Data specification (Input to Generator will be a 2-d tensor (noise/latent vector) and output will be the image again a 2d tensor and reverse it for discriminator however its output will be 1/0)

2) Cost function (generator: maximize the likelihood of discriminator classifying fake as real. discriminator: want to maximize the likelihood of classifying fake as fake and real as real)

3) Optimization procedure (adam,sgd)

4) Model (Design decisions : no. of layers, neurons in each, activation func(hidden+output) many more...)

# Activation function choice
1) tanh() is chosen first for the generator because its bounded, symmetric output works well for image generation, based on practical success and theoretical benefits.

2) Normalization follows to match the real data (EMNIST) to the generator’s output range of [-1, 1], ensuring the discriminator learns properly.

# BCEloss
BCE=−[y⋅log⁡(p)+(1−y)⋅log⁡(1−p)] here, y : label 0 or 1 and p: prob that the label is true

# In context of GAN
For Discriminator: first you find the loss for real images: now for that the BCEloss reduces to : BCE=−[y⋅log⁡(p)] (Note: if p is higher like 0.9 then log(p) would give negative higher number but as we are minimizing taking negative makes it smaller relatively. Hence, less loss.)

second you find loss for fake images: now the BCE reduces to: BCE=−[(1)⋅log⁡(1−p)] (Note: here also p means the same. The lesser it is the lesser the loss)

For Generator: Generator wants to fool discriminator into classifying fake as real(1).So, the BCE in this case will be: BCE=−[y⋅log⁡(p)]. Here p = D(G(z)). Higher the p lesser will be the loss.

# Improvement:

Also if generator seems powerful then adding a dropout might makes sense.

Label smoothing: To not completely eliminate the other half of loss function


