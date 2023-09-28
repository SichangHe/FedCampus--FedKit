from fed_kit import *

tflite_file = "pmdata.tflite"
coreml_file = "pmdata.mlmodel"
name = "PMData_unified"
tflite_layers = [14336, 2048, 1048576, 2048, 2048, 4]
coreml_layers = [
    {"name": "sequential/dense/BiasAdd", "type": "weights", "updatable": True},
    {"name": "sequential/dense/BiasAdd", "type": "bias", "updatable": True},
    {"name": "sequential/dense_1/BiasAdd", "type": "weights", "updatable": True},
    {"name": "sequential/dense_1/BiasAdd", "type": "bias", "updatable": True},
    {"name": "Identity", "type": "weights", "updatable": True},
    {"name": "Identity", "type": "bias", "updatable": True},
]
data_type = "PMData_7_1"
response = upload(
    tflite_file, coreml_file, name, tflite_layers, coreml_layers, data_type
)
if response.status_code < 200 or response.status_code >= 300:
    print(response.text)
    exit(1)
print("Successfully uploaded the unified PMData model.")
