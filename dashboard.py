import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# konfigurasi halaman dashboard
st.set_page_config(
    page_title="Dashboard Kualitas Udara Beijing",
    page_icon="☁",
    layout="wide"
)
st.title("Dashboard Analisis Kualitas Udara Beijing (2013–2017)")
st.markdown(
    "Dashboard ini menyajikan eksplorasi data, tren musiman, pengaruh"
    " meteorologi, serta analisis lanjutan berdasarkan data kualitas udara dari"
    " 12 stasiun di Beijing."
)

# fungsi load dataset
@st.cache_data
def load_data():
    file_id= "1h_0SQpi2EZqGtGNTAbv87HMSSDodFaym"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    # url= f"https://drive.google.com/file/d/1h_0SQpi2EZqGtGNTAbv87HMSSDodFaym/view?usp=sharing"
    df=pd.read_csv(url)
    # df=pd.read_csv("data/all_data_df.csv")
    if "year" not in df.columns and "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["year"] = df["datetime"].dt.year
    return df


try:
    df = load_data()
except Exception as e:
    st.error(
        f"Gagal memuat dataset. Pastikan file 'all_data.csv' ada di folder yang "
        f"sama. Detail error: {e}"
    )
    st.stop()

st.sidebar.header("Filter")
all_stations = sorted(df["station"].unique().tolist())
selected_stations = st.sidebar.multiselect(
    "Pilih Stasiun:", options=all_stations, default=all_stations
)

all_years = sorted(df["year"].unique().tolist())
selected_years = st.sidebar.multiselect(
    "Tahun:", options=all_years, default=all_years
)

filtered_df = df[
    df["station"].isin(selected_stations) & df["year"].isin(selected_years)
]


# Validasi jika filter kosong
if filtered_df.empty:
    st.warning(
        "⚠️ Data kosong! Silakan pilih minimal satu stasiun dan satu tahun"
        " pada sidebar."
    )
else:
    st.info(
        f"Menampilkan data dari **{len(selected_stations)} stasiun** dan"
        f" **{len(selected_years)} tahun** (Total baris data aktif:"
        f" {len(filtered_df):,})"
    )
    col1, col2 = st.columns(2)

    polutan_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    with col1:
        st.subheader("1. Peringkat Stasiun (Total Polutan)")
        station_avg = (
            filtered_df.groupby("station")[polutan_cols]
            .mean()
            .mean(axis=1)
            .sort_values(ascending=False)
        )

        fig1, ax1 = plt.subplots(figsize=(8, 4))
        colors = [
            "red" if i == 0 else ("green" if i == len(station_avg) - 1 else "skyblue")
            for i in range(len(station_avg))
        ]
        station_avg.plot(kind="bar", color=colors, ax=ax1, edgecolor="black")
        ax1.set_ylabel("Rata-rata Konsentrasi")
        ax1.set_xlabel("Stasiun")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig1)

    # --- CHART 2: RQ2 (Fluktuasi Musim & Dominasi CO) ---
    with col2:
        st.subheader("2. Fluktuasi Musim & Dominasi CO (Twin Axes)")
        if "season" in filtered_df.columns:
            df_season = filtered_df.groupby("season")[polutan_cols].mean()
            urutan_musim = ["spring", "summer", "autumn", "winter"]
            urutan_musim = [
                m for m in urutan_musim if m in df_season.index
            ] or list(df_season.index)
            df_season = df_season.reindex(urutan_musim)

            fig2, ax_left = plt.subplots(figsize=(8, 4))
            color = "tab:red"
            ax_left.set_ylabel("Konsentrasi CO", color=color, fontweight="bold")
            ax_left.plot(
                df_season.index,
                df_season["CO"],
                color=color,
                marker="o",
                linewidth=2,
                label="CO",
            )
            ax_left.tick_params(axis="y", labelcolor=color)

            ax_right = ax_left.twinx()
            for col in ["PM2.5", "PM10", "SO2", "NO2", "O3"]:
                if col in df_season.columns:
                    ax_right.plot(
                        df_season.index,
                        df_season[col],
                        marker="o",
                        linewidth=1.5,
                        label=col,
                    )
            ax_right.set_ylabel("Polutan Lain", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig2)
        else:
            st.warning("Kolom 'season' tidak ditemukan di dataset.")

    st.markdown("---")

    col3, col4 = st.columns(2)

    # --- CHART 3: RQ3 (Korelasi Meteorologi / Heatmap) ---
    with col3:
        st.subheader("3. Korelasi Faktor Cuaca & Polutan (Heatmap)")
        faktor_cuaca = ["RAIN", "WSPM", "TEMP", "PRES", "DEWP"]
        avail_cuaca = [c for c in faktor_cuaca if c in filtered_df.columns]
        avail_polutan = [c for c in polutan_cols if c in filtered_df.columns]

        if avail_cuaca and avail_polutan:
            subset_corr = (
                filtered_df[avail_cuaca + avail_polutan]
                .corr()
                .loc[avail_cuaca, avail_polutan]
            )
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            sns.heatmap(
                subset_corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                ax=ax3,
                linewidths=0.5,
                annot_kws={"size": 8},
            )
            plt.tight_layout()
            st.pyplot(fig3)

    # --- CHART 4: Advanced Analysis (Binning PM2.5) ---
    with col4:
        st.subheader("4. Analisis Lanjutan: Binning Kategori PM2.5")
        bins_pm25 = [0, 35, 75, 115, 150, 500]
        labels_pm25 = [
            "Baik",
            "Sedang",
            "Tidak Sehat",
            "Sangat Tidak Sehat",
            "Berbahaya",
        ]

        temp_df = filtered_df.copy()
        temp_df["Kategori_PM2.5"] = pd.cut(
            temp_df["PM2.5"], bins=bins_pm25, labels=labels_pm25
        )
        frekuensi_pm25 = temp_df["Kategori_PM2.5"].value_counts(sort=False)

        fig4, ax4 = plt.subplots(figsize=(8, 4))
        ax4 = frekuensi_pm25.plot(
            kind="bar", color="skyblue", edgecolor="black", ax=ax4
        )
        ax4.set_ylabel("Jumlah Kejadian (Jam)")
        plt.xticks(rotation=0, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig4)
