from dataclasses import dataclass


@dataclass(frozen=True)
class SiglipPreset:
    repo: str  # repositório no Hugging Face (modelos ONNX, Apache-2.0)
    folder: str  # pasta dentro de GRAPH_OCR_SIGLIP_MODELS_DIR
    size: int  # lado do quadrado que entra no modelo
    vision: str = "onnx/vision_model.onnx"
    text: str = "onnx/text_model.onnx"
    extra: tuple[str, ...] = ()  # arquivos que precisam estar ao lado (pesos externos)


SIGLIP_PRESETS = {
    "base": SiglipPreset("onnx-community/siglip2-base-patch16-224-ONNX", "siglip2-base-fp32", 224),
    # text_model.onnx do so400m guarda os pesos num arquivo externo de 2,8 GB
    "so400m": SiglipPreset(
        "onnx-community/siglip2-so400m-patch14-384-ONNX", "siglip2-so400m-fp32", 384, extra=("onnx/text_model.onnx_data",)
    ),
}
