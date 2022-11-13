from collections.abc import List
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import keyboard_simulator


class KeyboardDecoder(nn.Module) :
    """Simple keyboard decoder
    """
    layer_sizes: List[int]
    output_size: int

    def setup(self):
        self.layers = [nn.Dense(features=ms) for ms in self.layer_sizes]
        output_layer = nn.Dense(self.output_size)

    def __call__(self, x):
        for lr in self.layers:
            x = lr(x)
            x = nn.relu(x)
        return self.output_layer(x)

def BuildModel():
    """Builds a model and initialize parameters."""
    model = KeyboardDecoder(layer_sizes=[128, 128, 128], output_size=27)
    rng = jax.random.PRNGKey(42)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (2)) 
    params = model.init(init_rng, inp)
    return (model, params)


def BuildOptimizer():
    """Creates an optimizer"""
    return optax.adam(learning_rate=0.01)


def calculate_loss_accuracy(state, params, batch):
    data_input, labels = batch
    logits = state.apply_fn(params, data_input)
    labels_onehot = jax.nn.one_hot(labels, num_classes=keyboard_simulator.NUM_CLASSES)
    loss = optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return (loss, accuracy)


@jax.jit  # Jit the function for efficiency
def train_step(state, batch):
    # Gradient function
    grad_fn = jax.value_and_grad(calculate_loss_accuracy,  # Function to calculate the loss
                                 argnums=1,  # Parameters are second argument of the function
                                 has_aux=True  # Function has additional outputs, here accuracy
                                )
    # Determine gradients for current model, parameters and batch
    (loss, acc), grads = grad_fn(state, state.params, batch)
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    return state, loss, acc


@jax.jit  # Jit the function for efficiency
def eval_step(state, batch):
    # Determine the accuracy
    _, acc = calculate_loss_acc(state, state.params, batch)
    return acc

