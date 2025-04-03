import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Dicionário que mapeia índices para nomes dos Pokémon
class_names = {
    0: "Bulbasaur",
    1: "Charmander",
    2: "Eevee",
    3: "Pikachu",
    4: "Squirtle"
}

# Carregar o modelo treinado
model = tf.keras.models.load_model("pokemon_classifier.keras")

# Função para fazer a previsão de uma nova imagem
def predict_image(img_path, model, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=(target_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona dimensão do batch
    img_array /= 255.0  # Normalização (se foi usada no treinamento)

    # Fazer a previsão
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)  # Pega a classe com maior probabilidade
    confidence = np.max(predictions)  # Pega a confiança da previsão

    return class_index, confidence

# Teste com uma imagem específica
img_path = "z:/Sei la 4/ATIVIDADES EAD/EAD - IFTM - 8° Periodo/IA/GottaClassifyEmAll/test_images/bulba.jpg"  # Substitua pelo caminho da imagem de teste
class_index, confidence = predict_image(img_path, model)

# Obter nome do Pokémon
pokemon_name = class_names.get(class_index, "Desconhecido")

# Exibir resultados
print(f"Classe prevista: {class_index} ({pokemon_name}), Confiança: {confidence:.2f}")
