import tensorflow as tf
import tf2onnx
from mltu.tensorflow.losses import CTCloss  # Importa seu loss customizado, se necessário

# Carrega o modelo Keras. Caso haja objetos customizados (como CTCloss), passe-os em custom_objects.
model = tf.keras.models.load_model(
    "C:/Users/GELSON JUNIOR/Documents/GitHub/mltu/Models/1_image_to_word/202503131047/model.h5", 
    custom_objects={"CTCloss": CTCloss},
    compile=False
)

# Exiba o summary para confirmar que o modelo foi carregado corretamente
model.summary()

# Crie um input signature com base na forma de entrada do modelo.
# Supondo que o modelo possua apenas uma entrada:
spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name=model.inputs[0].name),)

# Defina o caminho de saída para o arquivo ONNX
output_path = "C:/Users/GELSON JUNIOR/Documents/GitHub/mltu/Models/1_image_to_word/202503131047/model.onnx"

# Realize a conversão
model_proto, external_tensor_storage = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)

print(f"Modelo ONNX salvo em: {output_path}")