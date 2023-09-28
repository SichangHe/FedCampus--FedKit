from .. import keras as k


def build_model():
    model = k.Sequential()
    for _ in range(2):
        model.add(
            k.layers.Dense(units=512, activation="relu", input_dim=7)  # type: ignore
        )

    model.add(k.layers.Dense(1))
    learning_rate = 0.0013826
    model.compile(
        optimizer=k.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model
