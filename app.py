import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Dashboard RFM", layout="wide")
st.title("🎯 Dashboard Analisis Segmentasi Pelanggan (RFM)")

@st.cache_data
def muat_data():
    # Pastikan file CSV ini memiliki kolom: Customer ID, Recency, Frequency, Monetary, dan Klaster
    df = pd.read_csv("data_pelanggan_tersegmentasi.xls")
    return df

df_rfm = muat_data()

st.sidebar.header("Navigasi")
pilihan_menu = st.sidebar.selectbox(
    "Pilih Halaman", 
    ["Ringkasan Data", "Analisis Klaster", "Simulasi Pelanggan Baru", "Rekomendasi Strategi"]
)

# ---------------------------------------------------------
# 1. HALAMAN Ringkasan DATA (Dengan Filter Interaktif)
# ---------------------------------------------------------
if pilihan_menu == "Ringkasan Data":
    st.subheader("📊 Karakteristik Data Pelanggan")
    
    # Fitur Interaktif: Filter dengan Slider
    st.markdown("**Filter Data Berdasarkan Kriteria RFM:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        min_r, max_r = int(df_rfm['Recency'].min()), int(df_rfm['Recency'].max())
        filter_r = st.slider("Recency (Hari)", min_r, max_r, (min_r, max_r))
    with col2:
        min_f, max_f = int(df_rfm['Frequency'].min()), int(df_rfm['Frequency'].max())
        filter_f = st.slider("Frequency (Transaksi)", min_f, max_f, (min_f, max_f))
    with col3:
        min_m, max_m = float(df_rfm['Monetary'].min()), float(df_rfm['Monetary'].max())
        filter_m = st.slider("Monetary (Total Belanja)", min_m, max_m, (min_m, max_m))

    # Terapkan filter
    df_filtered = df_rfm[
        (df_rfm['Recency'].between(filter_r[0], filter_r[1])) &
        (df_rfm['Frequency'].between(filter_f[0], filter_f[1])) &
        (df_rfm['Monetary'].between(filter_m[0], filter_m[1]))
    ]
    
    st.write(df_filtered.head(10))
    st.metric("Total Pelanggan Sesuai Filter", len(df_filtered))

# ---------------------------------------------------------
# 2. HALAMAN ANALISIS KLASTER
# ---------------------------------------------------------
elif pilihan_menu == "Analisis Klaster":
    st.subheader("📈 Visualisasi Karakteristik Klaster")
    opsi_fitur = st.selectbox("Pilih Metrik untuk Divisualisasikan", ["Recency", "Frequency", "Monetary"])

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x="Klaster", y=opsi_fitur, data=df_rfm, ax=ax, hue="Klaster", palette="Set2", legend=False)
    plt.xticks(rotation=0)
    plt.title(f'Distribusi {opsi_fitur} per Klaster')
    st.pyplot(fig)

# ---------------------------------------------------------
# 3. HALAMAN SIMULASI PELANGGAN BARU (Testing/Uji Coba)
# ---------------------------------------------------------
elif pilihan_menu == "Simulasi Pelanggan Baru":
    st.subheader("🧪 Uji Coba: Prediksi Klaster Pelanggan Hipotetis")
    st.markdown("Masukkan data perilaku pelanggan baru untuk memprediksi mereka akan masuk ke klaster mana.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        input_r = st.number_input("Recency (Hari sejak beli terakhir):", min_value=0, value=10)
    with col2:
        input_f = st.number_input("Frequency (Jumlah transaksi):", min_value=1, value=5)
    with col3:
        input_m = st.number_input("Monetary (Total belanja):", min_value=1.0, value=500.0)
        
    if st.button("Prediksi Klaster"):
        # Hitung rata-rata (centroid) dari masing-masing klaster dari data historis
        centroids = df_rfm.groupby('Klaster')[['Recency', 'Frequency', 'Monetary']].mean()
        
        # Hitung jarak Euclidean dari input ke setiap centroid klaster
        distances = {}
        for index, row in centroids.iterrows():
            dist = np.sqrt(
                (input_r - row['Recency'])**2 + 
                (input_f - row['Frequency'])**2 + 
                (input_m - row['Monetary'])**2
            )
            distances[index] = dist
            
        # Dapatkan klaster dengan jarak terdekat
        predicted_cluster = min(distances, key=distances.get)
        
        st.success(f"Berdasarkan pola historis, pelanggan ini diprediksi masuk ke: **{predicted_cluster}**")

# ---------------------------------------------------------
# 4. HALAMAN REKOMENDASI STRATEGI
# ---------------------------------------------------------
elif pilihan_menu == "Rekomendasi Strategi":
    st.subheader("💡 Strategi Pemasaran Berdasarkan Klaster")
    st.info("Rekomendasi Strategi:")
    st.markdown("""
    * 🌟 **High-Value / VIP:** Pertahankan loyalitas mereka dengan program keanggotaan eksklusif, penawaran premium VIP, dan promo pre-release.
    * 🌱 **Potensial / Loyal:** Tingkatkan interaksi mereka dengan mengirimkan rekomendasi produk yang personal dan program loyalti berinsentif (upselling).
    * ⚠️ **Berisiko:** Tawarkan kampanye win-back khusus, survei kepuasan, dan diskon bernilai tinggi untuk menarik mereka kembali bertransaksi.
    """)
