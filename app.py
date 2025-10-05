"""
╔═══════════════════════════════════════════════════════════╗
║       SEVENX MAX - API FINAL (PRONTA PARA DEPLOY)          ║
║  Modificações:                                           ║
║  - Otimizada para ser colocada no ar com PythonAnywhere. ║
║  - Carrega os dois modelos: Pico Pro (Texto) e Artista.  ║
╚═══════════════════════════════════════════════════════════╝
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import re
import unicodedata
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import MultiHeadAttention
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64

# --- Cria o objeto da aplicação Flask ---
# A variável 'app' precisa ser acessível globalmente para o PythonAnywhere
app = Flask(__name__)

# --- CONFIGURAÇÃO E CARREGAMENTO ---
# Caminhos relativos para funcionar no servidor
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_MODEL_DIR = os.path.join(BASE_DIR, "model/sevenx_tf_pico_pro")
TEXT_MODEL_PATH = os.path.join(TEXT_MODEL_DIR, "model.keras")
TOKENIZER_PATH = os.path.join(TEXT_MODEL_DIR, "tokenizer_vocab.pkl")
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "model/sevenx_artista_pico/generator.keras")

MAX_LEN = 35
VOCAB_SIZE = 10000
NOISE_DIM = 100

# --- Variáveis Globais ---
text_model = None
tokenizer_layer = None
index_to_word = None
image_generator = None

# ===================== CLASSES CUSTOMIZADAS (PARA CARREGAR O MODELO DE TEXTO) =====================
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim)])
        self.layernorm_1 = layers.LayerNormalization(); self.layernorm_2 = layers.LayerNormalization()
    def call(self, inputs, mask=None):
        attn_out = self.attention(query=inputs, value=inputs, key=inputs)
        proj_in = self.layernorm_1(inputs + attn_out)
        proj_out = self.dense_proj(proj_in)
        return self.layernorm_2(proj_in + proj_out)

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        return self.token_embeddings(inputs) + self.position_embeddings(positions)

# ===================== FUNÇÕES DE CARREGAMENTO DOS MODELOS =====================
def load_models():
    global text_model, tokenizer_layer, index_to_word, image_generator
    try:
        print("🧠 Carregando o modelo de TEXTO (Pico Pro)...")
        custom_objects = {"TransformerEncoder": TransformerEncoder, "PositionalEmbedding": PositionalEmbedding}
        text_model = keras.models.load_model(TEXT_MODEL_PATH, custom_objects=custom_objects)
        with open(TOKENIZER_PATH, 'rb') as f: vocab = pickle.load(f)
        tokenizer_layer = keras.layers.TextVectorization(max_tokens=VOCAB_SIZE, output_sequence_length=MAX_LEN, vocabulary=vocab)
        index_to_word = {i: w for i, w in enumerate(vocab)}
        print("✅ Modelo de TEXTO carregado!")
    except Exception as e:
        print(f"❌ Falha ao carregar o modelo de TEXTO: {e}")

    try:
        print("\n🎨 Carregando a rede neural 'Artista' (Imagem)...")
        image_generator = keras.models.load_model(IMAGE_MODEL_PATH)
        print("✅ Modelo de IMAGEM carregado!")
    except Exception as e:
        print(f"❌ Falha ao carregar o modelo de IMAGEM: {e}")

# ===================== FUNÇÕES DE GERAÇÃO =====================
def clean_text_for_generation(text):
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return f"[start] {text}"

def generate_text_response(prompt):
    if not text_model: return "O modelo de texto não foi carregado."
    cleaned_prompt = clean_text_for_generation(prompt)
    input_tokens = tokenizer_layer([cleaned_prompt])
    output_tokens = ["[start]"]
    for i in range(MAX_LEN - 1):
        decoder_input = tokenizer_layer([" ".join(output_tokens)])
        preds = text_model.predict([input_tokens, decoder_input], verbose=0)
        next_token_index = np.argmax(preds[0, i, :])
        next_word = index_to_word.get(next_token_index, "[unk]")
        if next_word == "[end]": break
        output_tokens.append(next_word)
    return " ".join(output_tokens[1:]).replace("[unk]", "").strip()

def generate_image_response():
    if not image_generator: return None
    noise = tf.random.normal([1, NOISE_DIM])
    generated_image = image_generator(noise, training=False)
    img_array = (generated_image[0] * 127.5 + 127.5).numpy().astype(np.uint8)
    img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ===================== ROTAS DA API =====================
@app.route('/')
def home(): return render_template('index.html')

@app.route('/api/generate_text', methods=['POST'])
def api_generate_text():
    prompt = request.json.get('prompt', '')
    if not prompt: return jsonify({'error': 'O prompt é obrigatório.'}), 400
    completion = generate_text_response(prompt)
    return jsonify({'success': True, 'completion': completion})

@app.route('/api/generate_image', methods=['POST'])
def api_generate_image():
    image_b64 = generate_image_response()
    if image_b64:
        return jsonify({'success': True, 'image': image_b64})
    return jsonify({'error': 'Não foi possível gerar a imagem.'}), 500

# --- CARREGAMENTO INICIAL ---
load_models()