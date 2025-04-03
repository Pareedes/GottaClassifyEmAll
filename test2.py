import os
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

# Indice de cada pokemon
class_names = {
    0: "Bulbasaur",
    1: "Charmander",
    2: "Eevee",
    3: "Pikachu",
    4: "Squirtle"
}

model = tf.keras.models.load_model("pokemon_classifier.keras")

def predict_image(img_path, model, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=(target_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)  # Pega a classe com maior probabilidade
    confidence = np.max(predictions)  # Pega a confiança da previsão

    return class_index, confidence

def save_result_image(img_path, pokemon_name, confidence, output_dir="result_images"):
    img = Image.open(img_path)

    if img.mode != "RGB":
        img = img.convert("RGB")

    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype("arial.ttf", 50)

    text = f"{pokemon_name} ({confidence:.2f}%)"

    text_position = (10, 10)

    text_color = "white"
    outline_color = "black"

    x, y = text_position
    draw.text((x-1, y-1), text, font=font, fill=outline_color)
    draw.text((x+1, y-1), text, font=font, fill=outline_color)
    draw.text((x-1, y+1), text, font=font, fill=outline_color)
    draw.text((x+1, y+1), text, font=font, fill=outline_color)
    draw.text(text_position, text, font=font, fill=text_color)

    os.makedirs(output_dir, exist_ok=True)

    # Salvar a imagem na pasta de resultados
    output_path = os.path.join(output_dir, os.path.basename(img_path))
    img.save(output_path)
    print(f"Imagem salva em: {output_path}")


img_name = input("Digite o nome da imagem: ")
img_path = os.path.join("test_images", img_name + ".jpg")

if not os.path.exists(img_path):
    print(f"A imagem '{img_name}.jpg' não foi encontrada")
else:
    class_index, confidence = predict_image(img_path, model)
    pokemon_name = class_names.get(class_index, "Desconhecido")
    print(f"Classe prevista: {class_index} ({pokemon_name}), Confiança: {confidence:.2f}%")

    # Salvar a imagem com o texto sobreposto
    save_result_image(img_path, pokemon_name, confidence)