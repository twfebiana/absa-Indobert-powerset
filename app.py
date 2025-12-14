# app.py
import streamlit as st
import pandas as pd
# from utils.inference import predict_single, predict_batch_dataframe
from utils import inference
from utils.visualization import plot_sentiment_counts, plot_aspect_freq
import io

@st.cache_resource
def init_model():
    inference.load_model()
    return True

init_model()

st.set_page_config(page_title="Aplikasi Sentimen & Aspek", layout="wide")

st.title("Aplikasi Analisis Sentimen & Aspek")
st.sidebar.title("Menu")
page = st.sidebar.radio("Pilih Halaman", ["Home", "Prediksi Ulasan Tunggal", "Prediksi CSV"])

if page == "Home":
    st.markdown("""
    **Deskripsi**  
    Aplikasi ini menerima *single review* atau *CSV* (kolom `ulasan`) untuk:
    - Memprediksi label powerset 
    - Memprediksi sentimen pada aspek (Kualitas Pelayanan Medis dan Staf (kpms), Fasilitas dan Infrastruktur (fi), Waktu Tunggu (wt), Biaya Layanan(bl))
    - Menampilkan grafik distribusi sentimen dan aspek
    """)

if page == "Prediksi Ulasan Tunggal":
    st.header("Prediksi Ulasan")
    text = st.text_area("Masukkan ulasan", height=150)
    if st.button("Prediksi"):
        if not text.strip():
            st.warning("Tolong masukkan teks ulasan.")
        else:
            with st.spinner("Melakukan prediksi..."):
                # res = predict_single(text)
                res = inference.predict_single(text)
            st.success("Selesai")
            st.write("**Teks:**", res["clean_text"])
            st.write("**Kelas Prediksi:**", res["class"])
            st.write("**Kombinasi Biner:**", res["binary"])
            st.write("**Aspek (Kualitas Pelayanan Medis dan Staf (kpms), Fasilitas dan Infrastruktur (fi), Waktu Tunggu (wt), Biaya Layanan (bl)):**")
            st.json(res["aspects"])
            # st.write("**Probabilitas:**")
            # st.write(res["probs"])

if page == "Prediksi CSV":
    st.header("Prediksi CSV")
    uploaded = st.file_uploader("Masukkan file CSV (Pastikan CSV mengandung kolom bernama 'ulasan')", type=["csv"])
    if uploaded is not None:
        df_uploaded = pd.read_csv(uploaded)
        if "ulasan" not in df_uploaded.columns:
            st.error("CSV harus mengandung kolom bernama 'ulasan'")
        elif not df_uploaded["ulasan"].astype(str).apply(
            lambda x: x.startswith('"') and x.endswith('"')
        ).all():
            st.error(
                'Setiap ulasan harus diapit tanda petik ganda (" ").\n\n'
                'Contoh yang benar:\n'
                '"Pelayanan suster ramah dan cepat"'
            )
        else:
            st.write("Preview Data:")
            st.dataframe(df_uploaded.head())

            if st.button("Proses CSV"):
                with st.spinner("Melakukan prediksi data..."):
                    # pred_df = predict_batch_dataframe(df_uploaded, text_col="ulasan")
                    pred_df = inference.predict_batch_dataframe(df_uploaded, text_col="ulasan")
                    for a in ["kpms","fi","wt","bl"]:
                        if f"aspect_{a}" not in pred_df.columns:
                            pred_df[f"aspect_{a}"] = 0
                    # plot
                    fig1, pred_df2 = plot_sentiment_counts(pred_df)
                    fig2 = plot_aspect_freq(pred_df)
                st.subheader("Hasil Prediksi (preview)")
                st.dataframe(pred_df.head(50))

                st.subheader("Distribusi Sentimen")
                st.pyplot(fig1)

                st.subheader("Frekuensi Aspek")
                st.pyplot(fig2)

                # download
                csv_bytes = pred_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download hasil prediksi (CSV)", data=csv_bytes, file_name="prediksi_hasil.csv", mime="text/csv")
