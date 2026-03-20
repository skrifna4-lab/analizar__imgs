from flask import Flask, request, jsonify
from PIL import Image
import torch

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    ViltProcessor,
    ViltForQuestionAnswering
)

app = Flask(__name__)

device = "cpu"

# ==============================
# 🔹 BLIP (Siempre cargado)
# ==============================
print("🔄 Cargando BLIP...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model.to(device)
print("✅ BLIP listo")

# ==============================
# 🔹 ViLT (lazy loading 🔥)
# ==============================
vilt_processor = None
vilt_model = None

def load_vilt():
    global vilt_processor, vilt_model
    if vilt_model is None:
        print("⚡ Cargando ViLT...")
        vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        vilt_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        vilt_model.to(device)
        print("✅ ViLT listo")

# ==============================
# 🧠 ENDPOINT
# ==============================
@app.route("/vision", methods=["POST"])
def vision():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    image = Image.open(request.files["image"]).convert("RGB")

    question = request.form.get("question")

    # ==========================
    # 🔹 SOLO DESCRIPCIÓN
    # ==========================
    if not question:
        inputs = blip_processor(images=image, return_tensors="pt").to(device)
        output = blip_model.generate(**inputs)
        caption = blip_processor.decode(output[0], skip_special_tokens=True)

        return jsonify({
            "type": "description",
            "result": caption
        })

    # ==========================
    # 🔹 PREGUNTA (VQA)
    # ==========================
    load_vilt()

    inputs = vilt_processor(image, question, return_tensors="pt").to(device)
    outputs = vilt_model(**inputs)

    logits = outputs.logits
    idx = logits.argmax(-1).item()
    answer = vilt_model.config.id2label[idx]

    return jsonify({
        "type": "answer",
        "question": question,
        "result": answer
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
