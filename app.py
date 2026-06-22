# 1. Instalasi Library (Sangat cepat dan tanpa konflik versi)
!pip install -q gradio gensim pandas

import gradio as gr
import pandas as pd
from gensim.models import Word2Vec, FastText

# 2. Setup Data Default (Bisa diedit nanti di interface)
default_corpus = """
natural language processing atau nlp adalah cabang dari kecerdasan buatan.
sistem nlp menggunakan word embedding untuk merepresentasikan kata ke dalam bentuk vektor.
word2vec dan glove merupakan metode representasi vektor kata yang sangat populer.
fasttext dikembangkan oleh facebook ai research untuk menangani subword information.
glove didasarkan pada teknik matriks faktorisasi global dalam sebuah korpus teks.
model word2vec menggunakan algoritma cbow dan skip gram untuk melatih representasi kata.
kemiripan semantik antar kata dapat dihitung menggunakan rumus cosine similarity.
"""

# Variabel global untuk menyimpan model
global_model = None

# 3. Fungsi-fungsi Backend
def train_model(corpus, model_type, vector_size, window, min_count):
    global global_model
    # Preprocessing sederhana
    sentences = [sentence.lower().split() for sentence in corpus.split("\n") if sentence.strip()]
    
    if not sentences:
        return "Error: Korpus tidak boleh kosong!"
        
    if model_type == "Word2Vec":
        global_model = Word2Vec(sentences=sentences, vector_size=int(vector_size), window=int(window), min_count=int(min_count), epochs=20)
    else:
        global_model = FastText(sentences=sentences, vector_size=int(vector_size), window=int(window), min_count=int(min_count), epochs=20)
        
    vocab_size = len(global_model.wv.index_to_key)
    return f"✅ Sukses! Model {model_type} berhasil dilatih dengan {vocab_size} kosakata unik."

def find_similar(word, top_n):
    global global_model
    if global_model is None:
        return pd.DataFrame([["Latih model terlebih dahulu!", 0]]), "Model belum siap."
    try:
        results = global_model.wv.most_similar(word.lower(), topn=int(top_n))
        df = pd.DataFrame(results, columns=["Kata Serupa", "Skor Cosine Similarity"])
        # Mengambil 5 dimensi awal dari vektor kata tersebut
        vector_repr = str(global_model.wv[word.lower()][:5]) + " ... (dipotong)"
        return df, f"Representasi Vektor (5 dimensi awal):\n{vector_repr}"
    except Exception as e:
        return pd.DataFrame(), f"Error: Kata '{word}' tidak ditemukan di kosakata."

def calculate_similarity(word1, word2):
    global global_model
    if global_model is None:
        return "Latih model terlebih dahulu!"
    try:
        score = global_model.wv.similarity(word1.lower(), word2.lower())
        return f"Nilai Cosine Similarity antara '{word1}' dan '{word2}' adalah: {score:.4f}"
    except Exception as e:
        return f"Error: Salah satu atau kedua kata tidak ditemukan di kosakata."

# 4. Membangun Interface (UI) Menggunakan Gradio
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🤖 Interface Hasil Word Vector Representations")
    gr.Markdown("**Tugas Mata Kuliah Advanced NLP - Universitas Pamulang_MUH. ARMIL SYAM(241012050114)**")
    
    with gr.Tab("1. ⚙️ Latih Model"):
        gr.Markdown("Masukkan teks untuk melatih model Word2Vec atau FastText secara langsung.")
        corpus_input = gr.Textbox(lines=8, label="Dataset (Korpus Teks)", value=default_corpus)
        with gr.Row():
            model_dropdown = gr.Dropdown(["Word2Vec", "FastText"], label="Pilih Metode", value="Word2Vec")
            v_size = gr.Slider(10, 300, 100, step=10, label="Vector Size (Dimensi)")
            win_size = gr.Slider(2, 10, 5, step=1, label="Window Size")
            m_count = gr.Slider(1, 5, 1, step=1, label="Minimum Count")
        train_btn = gr.Button("🚀 Latih Model Sekarang", variant="primary")
        train_output = gr.Textbox(label="Status Pelatihan", interactive=False)
        
        train_btn.click(train_model, inputs=[corpus_input, model_dropdown, v_size, win_size, m_count], outputs=train_output)

    with gr.Tab("2. 🔍 Cari Kata Mirip (Most Similar)"):
        with gr.Row():
            target_word = gr.Textbox(label="Masukkan Kata Target (contoh: nlp)")
            top_n_slider = gr.Slider(1, 15, 5, step=1, label="Tampilkan Top-N")
        sim_btn = gr.Button("Cari Kata Paling Mirip")
        
        sim_output_table = gr.Dataframe(label="Tabel Hasil Kemiripan")
        sim_output_vector = gr.Textbox(label="Nilai Matriks Vektor")
        
        sim_btn.click(find_similar, inputs=[target_word, top_n_slider], outputs=[sim_output_table, sim_output_vector])

    with gr.Tab("3. ⚖️ Kemiripan Pasangan Kata"):
        with gr.Row():
            word1 = gr.Textbox(label="Kata Pertama (A)")
            word2 = gr.Textbox(label="Kata Kedua (B)")
        pair_btn = gr.Button("Hitung Cosine Similarity")
        pair_output = gr.Textbox(label="Hasil Evaluasi Pasangan")
        
        pair_btn.click(calculate_similarity, inputs=[word1, word2], outputs=pair_output)

# 5. Menjalankan Interface dan Membuat Public Link
interface.launch(share=True, debug=True)
