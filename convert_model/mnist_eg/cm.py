from coremltools.models.neural_network import AdamParams, NeuralNetworkBuilder

from .. import keras as k
from ..coreml import (
    convert,
    nn_builder,
    random_fit,
    save_builder,
    try_make_layers_updatable,
)
from . import in_shape, mnist_sequence, n_classes

COREML_FILE = "mnist.mlmodel"


def config_builder(builder: NeuralNetworkBuilder):
    softmax_out_name = "SoftmaxLast_true"
    builder.add_softmax("SoftmaxLast", "Identity", softmax_out_name)
    builder.set_categorical_cross_entropy_loss("lossLayer", input=softmax_out_name)
    builder.set_adam_optimizer(AdamParams())
    max_epochs = 10
    builder.set_epochs(max_epochs, range(1, max_epochs + 1))


def main():
    model = k.Sequential(mnist_sequence())
    model.compile(loss="mse") # Just to make it fit once.
    random_fit(model, in_shape, (n_classes,))
    mlmodel = convert(model)
    builder = nn_builder(mlmodel)
    config_builder(builder)
    try_make_layers_updatable(builder, 2)
    builder.inspect_layers()
    save_builder(builder, COREML_FILE)


if __name__ == "__main__":
    main()
