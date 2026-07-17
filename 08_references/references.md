## ReLU (Activation Function)

ReLU stands for Rectified Linear Unit:

$$\text{ReLU}(x) = \max(0, x)$$

### Simple Behavior

- Positive values → kept  
- Negative values → set to 0  

Example:

ReLU(5.3) = 5.3

ReLU(-2.1) = 0.0

### Why It Matters

Without ReLU, multiple linear layers collapse into one.  
ReLU introduces non-linearity, allowing the model to learn complex patterns.

Used inside:
- Feed-Forward Networks (FFN)
- Many deep learning architectures