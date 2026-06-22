import streamlit as st
import gensim
from gensim.models import Word2Vec, FastText
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import plotly.express as px

# Setup Halaman
st.set_page_config(page_title="Word Vector App", page_icon="🤖", layout="wide")

st.title("🤖 Interface Hasil Word Vector Representations")
st.markdown("**Tugas Mata Kuliah Advanced NLP - Universitas Pamulang**")

# Default Corpus
default_corpus = """
natural language processing atau nlp adalah cabang dari kecerdasan buatan.
sistem nlp menggunakan word embedding untuk merepresentasikan kata ke dalam bentuk vektor.
word2vec dan glove merupakan metode representasi vektor kata yang sangat populer.
fasttext dikembangkan oleh facebook ai research untuk menangani subword information.
glove didasarkan pada teknik matriks faktorisasi global dalam sebuah korpus teks.
model word2vec menggunakan algoritma cbow dan skip gram untuk melatih representasi kata.
kemiripan semantik antar kata dapat dihitung menggunakan rumus cosine similarity.
komputer tidak memahami teks secara langsung melainkan memahami angka atau vektor data.
pembelajaran mendalam atau deep learning sangat bergantung pada representasi data ini.
"""

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Konfigurasi Model")
model_type = st.sidebar.selectbox("Metode Word Vector:", ["Word2Vec", "FastText"])
vector_size = st.sidebar.slider("Vector Size", 50, 300, 100, step=50)
window = st.sidebar.slider("Window Size", 2, 10, 5)
min_count = st.sidebar.number_input("Min Count", min_value=1, value=1)
epochs = st.sidebar.slider("Epochs", 5, 50, 20)

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Cara Penggunaan:**\n1. Masukkan teks korpus.\n2. Klik **Train Model**.\n3. Jelajahi tab hasil di sebelah kanan.")

# ================= MAIN AREA =================
corpus_input = st.text_area("📝 Masukkan Dataset / Korpus Teks Bahasa Indonesia:", value=default_corpus.strip(), height=150)

# Fungsi Training
@st.cache_resource
def train_model(m_type, corpus_text, v_size, win, min_c, eps):
    sentences = [sentence.lower().split() for sentence in corpus_text.split("\n") if sentence.strip()]
    if m_type == "Word2Vec":
        return Word2Vec(sentences=sentences, vector_size=v_size, window=win, min_count=min_c, epochs=eps)
    else:
        return FastText(sentences=sentences, vector_size=v_size, window=win, min_count=min_c, epochs=eps)

# Tombol Eksekusi
if st.button("🚀 Train Model Sekarang", type="primary"):
    if not corpus_input.strip():
        st.error("Korpus tidak boleh kosong!")
    else:
        with st.spinner("Sedang melatih model..."):
            st.session_state['model'] = train_model(model_type, corpus_input, vector_size, window, min_count, epochs)
            st.success(f"Model {model_type} berhasil dilatih!")

# ================= TABS HASIL =================
if 'model' in st.session_state:
    model = st.session_state['model']
    wv = model.wv
    vocab = list(wv.index_to_key)
    
    st.info(f"✅ **Model Aktif:** {model_type} | **Kosakata Unik:** {len(vocab)} kata")

    tab1, tab2, tab3 = st.tabs(["🔍 Cari Kata Mirip", "⚖️ Cosine Similarity", "📊 Visualisasi PCA (2D)"])

    # --- TAB 1 ---
    with tab1:
        st.subheader("Mencari Kata Terdekat (Most Similar)")
        col1, col2 = st.columns([2, 1])
        with col1:
            target_word = st.text_input("Ketik Kata (contoh: nlp):", value="nlp").lower()
        with col2:
            top_n = st.slider("Tampilkan Top N:", 1, 10, 5)
            
        if st.button("Cari Kemiripan"):
            try:
                hasil = wv.most_similar(target_word, topn=top_n)
                df = pd.DataFrame(hasil, columns=["Kata Serupa", "Skor Similarity"])
                df.index += 1
                st.dataframe(df, use_container_width=True)
            except Exception:
                st.error(f"Kata '{target_word}' tidak ada di kamus. Coba kata lain.")

    # --- TAB 2 ---
    with tab2:
        st.subheader("Hitung Kedekatan Dua Kata")
        colA, colB = st.columns(2)
        with colA:
            wordA = st.text_input("Kata Pertama:", value="sistem").lower()
        with colB:
            wordB = st.text_input("Kata Kedua:", value="komputer").lower()
            
        if st.button("Hitung Jarak Jaringan"):
            try:
                skor = wv.similarity(wordA, wordB)
                st.metric(label="Nilai Cosine Similarity", value=f"{skor:.4f}")
            except Exception:
                st.error("Pastikan kedua kata pernah ditulis di dalam korpus teks!")

    # --- TAB 3 ---
    with tab3:
        st.subheader("Peta Semantik Kata")
        if len(vocab) >= 3:
            vectors = np.array([wv[w] for w in vocab])
            pca = PCA(n_components=2)
            vecs_2d = pca.fit_transform(vectors)
            
            df_pca = pd.DataFrame(vecs_2d, columns=['X', 'Y'])
            df_pca['Kata'] = vocab
            
            fig = px.scatter(df_pca, x='X', y='Y', text='Kata')
            fig.update_traces(textposition='top center', marker=dict(size=12, color='#E67E22'))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Butuh minimal 3 kata unik untuk membuat grafik.")
