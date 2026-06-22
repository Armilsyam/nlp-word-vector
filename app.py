import streamlit as st
import gensim
from gensim.models import Word2Vec, FastText
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import plotly.express as px
import os

# Configuration Halaman Streamlit
st.set_page_config(
    page_title="Word Vector Representation UI",
    page_icon="🤖",
    layout="wide"
)

# Judul Utama
st.title("🤖 Interface Hasil Word Vector Representations")
st.caption("Dibuat untuk memenuhi Tugas Mata Kuliah Advanced NLP - Universitas Pamulang")

# Data teks contoh (Indonesian Corpus Mini)
default_corpus = """
natural language processing atau nlp adalah cabang dari kecerdasan buatan.
sistem nlp menggunakan word embedding untuk merepresentasikan kata ke dalam bentuk vektor.
word2vec dan glove merupakan metode representasi vektor kata yang sangat populer.
fasttext dikembangkan oleh facebook ai research untuk menangani subword information.
glove didasarkan pada teknik matriks faktorisasi global co-occurrence dalam sebuah korpus teks.
model word2vec menggunakan algoritma cbow dan skip gram untuk melatih representasi kata.
kemiripan semantik antar kata dapat dihitung menggunakan rumus cosine similarity.
komputer tidak memahami teks secara langsung melainkan memahami angka atau vektor data.
pembelajaran mendalam atau deep learning sangat bergantung pada kualitas representasi data ini.
proses nlp biasanya melalui tahap tokenisasi normalisasi dan stopword removal sebelum pemodelan.
"""

# ==========================================
# SIDEBAR: KONTROL DAN PARAMETER MODEL
# ==========================================
st.sidebar.header("⚙️ Pengaturan Model & Dataset")

# Pilihan Tipe Model
model_type = st.sidebar.selectbox(
    "Pilih Metode Word Vector:",
    ["Word2Vec", "FastText", "Load Pre-trained GloVe/Word2Vec"]
)

if model_type in ["Word2Vec", "FastText"]:
    st.sidebar.subheader("Hyperparameters Model")
    
    vector_size = st.sidebar.slider("Vector Size (Dimensi Vektor)", 50, 300, 100, step=50)
    window = st.sidebar.slider("Context Window Size", 2, 10, 5)
    min_count = st.sidebar.number_input("Minimum Word Count (min_count)", min_value=1, value=1)
    sg = st.sidebar.selectbox("Algoritma (sg)", ["CBOW (0)", "Skip-gram (1)"], index=1)
    sg_val = 1 if "Skip-gram" in sg else 0
    epochs = st.sidebar.slider("Jumlah Epochs", 5, 100, 20)

    st.subheader("📝 Dataset / Korpus Teks")
    corpus_input = st.text_area(
        "Masukkan teks/korpus bahasa Indonesia untuk training model:",
        value=default_corpus.strip(),
        height=180
    )
    
    tokenized_sentences = [sentence.lower().split() for sentence in corpus_input.split("\n") if sentence.strip()]

    @st.cache_resource
    def train_custom_model(m_type, sentences, v_size, win, min_c, algorithms, eps):
        if m_type == "Word2Vec":
            model = Word2Vec(sentences=sentences, vector_size=v_size, window=win, min_count=min_c, sg=algorithms, epochs=eps, workers=4)
        else:
            model = FastText(sentences=sentences, vector_size=v_size, window=win, min_count=min_c, sg=algorithms, epochs=eps, workers=4, min_n=3, max_n=6)
        return model

    if st.sidebar.button("🚀 Train Model Sekarang"):
        if not tokenized_sentences:
            st.error("Korpus teks tidak boleh kosong!")
        else:
            with st.spinner("Sedang melatih model word embedding..."):
                st.session_state['model'] = train_custom_model(model_type, tokenized_sentences, vector_size, window, min_count, sg_val, epochs)
                st.sidebar.success(f"Model {model_type} Berhasil Dilatih!")

else:
    st.sidebar.subheader("Load Pre-trained File")
    st.sidebar.warning("⚠️ Fitur ini hanya berfungsi jika file model dijalankan secara lokal, bukan di Streamlit Cloud.")
    glove_path = st.sidebar.text_input("Path File Word2Vec/GloVe (.txt)", value="C:/Users/Asus/python/glove.6B.300d.word2vec.txt")
    
    if st.sidebar.button("📂 Load Model"):
        if not os.path.exists(glove_path):
            st.sidebar.error(f"File tidak ditemukan di path: {glove_path}. Pastikan Anda menjalankannya di komputer lokal.")
        else:
            try:
                with st.spinner("Memuat model eksternal..."):
                    model = gensim.models.KeyedVectors.load_word2vec_format(glove_path, binary=False)
                    st.session_state['model'] = model
                    st.sidebar.success("Model Eksternal Berhasil Dimuat!")
            except Exception as e:
                st.sidebar.error(f"Gagal memuat file.\nError: {e}")


# ==========================================
# HALAMAN UTAMA: INTERFACE FITUR
# ==========================================
if 'model' in st.session_state:
    model = st.session_state['model']
    wv = model.wv if hasattr(model, 'wv') else model
    vocabulary = list(wv.index_to_key)

    col_info1, col_info2 = st.columns(2)
    col_info1.metric(label="Model Aktif", value=model_type)
    col_info2.metric(label="Jumlah Kosakata Unik (Vocabulary)", value=len(vocabulary))

    tab1, tab2, tab3 = st.tabs(["🔍 Cari Kata Mirip", "⚖️ Kemiripan Pasangan Kata", "📊 Visualisasi Ruang Vektor 2D"])

    # --- TAB 1 ---
    with tab1:
        st.header("Mencari Kata Terdekat (Semantic Proximity)")
        col_t1_1, col_t1_2 = st.columns([2, 1])
        with col_t1_1:
            if model_type == "FastText":
                target_word = st.text_input("Ketik Kata Target:", value="nlp").lower()
            else:
                target_word = st.selectbox("Pilih Kata Target dari Kosakata:", options=vocabulary)
        
        with col_t1_2:
            top_n = st.slider("Tampilkan Top N kata mirip:", 3, 20, 5)

        if st.button("Hitung Kemiripan Kata", key="btn_similar"):
            try:
                similar_words = wv.most_similar(target_word, topn=top_n)
                df_similar = pd.DataFrame(similar_words, columns=["Kata Serupa", "Skor Cosine Similarity"])
                df_similar.index = df_similar.index + 1
                
                st.subheader(f"Daftar Kata Paling Mirip dengan '{target_word}':")
                st.dataframe(df_similar.style.format({"Skor Cosine Similarity": "{:.4f}"}), use_container_width=True)
                
                st.subheader(f"Vektor Semantik '{target_word}' (10 Dimensi Pertama):")
                st.code(str(wv[target_word][:10]))
            except KeyError:
                st.error(f"Kata '{target_word}' tidak ditemukan di dalam kosakata.")

    # --- TAB 2 ---
    with tab2:
        st.header("Menghitung Nilai Kedekatan Dua Kata")
        col_t2_1, col_t2_2 = st.columns(2)
        with col_t2_1:
            if model_type == "FastText":
                word_a = st.text_input("Masukkan Kata Pertama (A):", value="sistem").lower()
            else:
                word_a = st.selectbox("Pilih Kata Pertama (A):", options=vocabulary, index=0)
        with col_t2_2:
            if model_type == "FastText":
                word_b = st.text_input("Masukkan Kata Kedua (B):", value="nlp").lower()
            else:
                word_b = st.selectbox("Pilih Kata Kedua (B):", options=vocabulary, index=min(1, len(vocabulary)-1))

        if st.button("Hitung Cosine Similarity", key="btn_pair"):
            try:
                similarity_score = wv.similarity(word_a, word_b)
                st.info(f"Hasil Analisis Kedekatan Kosakata:")
                st.metric(label=f"Cosine Similarity antara '{word_a}' dan '{word_b}'", value=f"{similarity_score:.4f}")
            except KeyError as e:
                st.error(f"Salah satu kata tidak ditemukan dalam kamus indeks model: {e}")

    # --- TAB 3 ---
    with tab3:
        st.header("Visualisasi Hubungan Kata dalam Ruang Vektor 2D")
        
        if len(vocabulary) >= 3:
            word_vectors = np.array([wv[word] for word in vocabulary])
            pca = PCA(n_components=2)
            vectors_2d = pca.fit_transform(word_vectors)
            
            df_visual = pd.DataFrame(vectors_2d, columns=['Komponen X', 'Komponen Y'])
            df_visual['Kata'] = vocabulary
            
            fig = px.scatter(
                df_visual, x='Komponen X', y='Komponen Y', text='Kata',
                title=f"Peta Klaster Semantik Kata (Metode {model_type})"
            )
            fig.update_traces(textposition='top center', marker=dict(size=10, color='royalblue'))
            fig.update_layout(height=600, template="plotly_white")
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Jumlah kosakata dalam korpus teks latihan Anda minimal harus berisi 3 kata untuk visualisasi PCA.")
else:
    st.info("💡 **Petunjuk Penggunaan:** Silakan pilih konfigurasi model di bilah sisi kiri, kemudian klik tombol **'Train Model Sekarang'**.")
