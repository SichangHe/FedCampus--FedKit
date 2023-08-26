from django.urls import path
from train.views import (
    advertise_model,
    request_server,
    store_params,
    upload_coreml,
    upload_tflite,
    which_coreml,
)

urlpatterns = [
    path("advertised", advertise_model),
    path("which_coreml", which_coreml),
    path("server", request_server),
    path("upload", upload_tflite),
    path("upload_coreml", upload_coreml),
    path("params", store_params),
]
