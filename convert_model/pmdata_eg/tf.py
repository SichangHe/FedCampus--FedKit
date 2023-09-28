from ..tflite import (
    SAVED_MODEL_DIR,
    BaseTFLiteModel,
    convert_saved_model,
    save_model,
    save_tflite_model,
    tflite_model_class,
)
from . import in_dim, build_model

TFLITE_FILE = "pmdata.tflite"


@tflite_model_class
class PMDataTFModel(BaseTFLiteModel):
    X_SHAPE = [in_dim]
    Y_SHAPE = [1]

    def __init__(self):
        self.model = build_model()


def tflite():
    model = PMDataTFModel()
    save_model(model, SAVED_MODEL_DIR)
    tflite_model = convert_saved_model(SAVED_MODEL_DIR)
    save_tflite_model(tflite_model, TFLITE_FILE)


if __name__ == "__main__":
    tflite()
