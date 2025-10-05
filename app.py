"""
╔═══════════════════════════════════════════════════════════╗
║           SEVENX MAX - APP PARA MODELO 1M                 ║
║  Corrige deserialização do PositionalEmbedding e roda     ║
║  o modelo de conversa 1M diretamente no Flask            ║
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
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ===================== CLASSES PERSONALIZADAS =====================
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([layers.Dense(dense_dim, activation="relu"),
                                            layers.Dense(embed_dim)])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        
    def call(self, inputs, mask=None):
        attn_out = self.attention(query=inputs, value=inputs, key=inputs)
        proj_in = self.layernorm_1(inputs + attn_out)
        proj_out = self.dense_proj(proj_in)
        return self.layernorm_2(proj_in + proj_out)

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)  # aceita name, trainable, dtype
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        return self.token_embeddings(inputs) + self.position_embeddings(positions)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim
        })
        return config

# ===================== CONFIGURAÇÃO =====================
MODEL_DIR = "./model/sevenx_tf_conversa_1m"
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer_vocab.pkl")

MAX_LEN = 25
VOCAB_SIZE = 6000

model = None
tokenizer_layer = None
index_to_word = None

# ===================== FUNÇÕES =====================
def clean_text(text):
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'[^a-z0-9\s_.,?]', '', text)
    return f"[start] {text}"

def load_model_and_tokenizer():
    global model, tokenizer_layer, index_to_word
    print("🧠 Carregando modelo 1M...")
    custom_objects = {
        "TransformerEncoder": TransformerEncoder,
        "PositionalEmbedding": PositionalEmbedding
    }
    model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    with open(TOKENIZER_PATH, 'rb') as f:
        vocab = pickle.load(f)
    tokenizer_layer = layers.TextVectorization(max_tokens=VOCAB_SIZE, output_sequence_length=MAX_LEN, vocabulary=vocab)
    index_to_word = {i: w for i, w in enumerate(vocab)}
    print("✅ Modelo e tokenizer carregados com sucesso!")

def generate_response(prompt):
    if model is None:
        return "Modelo não carregado."
    
    cleaned_prompt = clean_text(prompt)
    input_tokens = tokenizer_layer([cleaned_prompt])
    output_tokens = ["[start]"]

    for i in range(MAX_LEN - 1):
        decoder_input = tokenizer_layer([" ".join(output_tokens)])
        preds = model.predict([input_tokens, decoder_input], verbose=0)
        next_token_index = np.argmax(preds[0, i, :])
        next_word = index_to_word.get(next_token_index, "[unk]")
        if next_word == "[end]": break
        output_tokens.append(next_word)
    
    return " ".join(output_tokens[1:]).replace("[unk]", "").strip()

# ===================== ROTAS =====================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.json
    prompt = data.get('prompt', '').strip()
    if not prompt:
        return jsonify({'error': 'O prompt é obrigatório.'}), 400
    completion = generate_response(prompt)
    return jsonify({'success': True, 'completion': completion})

# ===================== INICIAR FLASK =====================
if __name__ == '__main__':
    load_model_and_tokenizer()
    print("\n🚀 SEVENX MAX - API 1M rodando")
    app.run(host='127.0.0.1', port=5001, debug=False)
