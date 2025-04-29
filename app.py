import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pca import run_pca, plot_pca_2d, plot_pca_3d, plot_variance, plot_variance_bar, plot_loadings
from io import BytesIO
import map  # your map.py module

# Upload CSV
st.title("PCA on Garnet Data")
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Drop index if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Select features
    features = st.multiselect("Select Features", options=df.columns.tolist(), default=[
        'SiO2', 'Al2O3', 'MgO', 'TiO2', 'Cr2O3', 'CaO', 'MnO', 'Na2O', 'K2O', 'FeO'])

    # Optional: select target
    target_column = st.selectbox("Select Target (for coloring, optional)", options=['None'] + df.columns.tolist())

    if features:
        # Run PCA
        X_pca, pca, X_scaled, components = run_pca(df, features)
        target = df[target_column] if target_column != 'None' else None

        # Show DataFrame download
        pca_df = pd.DataFrame(X_pca, columns=[f"PC{i + 1}" for i in range(X_pca.shape[1])])
        if target_column != 'None':
            pca_df[target_column] = target.values

        csv = pca_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download PCA-transformed Data (CSV)", csv, "pca_results.csv", "text/csv")

        var_df = pd.DataFrame({
            "Principal Component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            "Explained Variance (%)": pca.explained_variance_ratio_ * 100
        })
        csv_var = var_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Explained Variance (CSV)", csv_var, "explained_variance.csv", "text/csv")


        # Helper to convert matplotlib figure to JPEG
        def fig_to_jpeg(fig):
            buf = BytesIO()
            fig.savefig(buf, format="jpeg", dpi=300, bbox_inches="tight")
            buf.seek(0)
            return buf


        # 2D plot
        fig1 = plot_pca_2d(X_pca, target)
        st.pyplot(fig1)
        st.download_button("Download 2D PCA Plot (JPEG)", fig_to_jpeg(fig1), file_name="pca_2d.jpeg")

        # 3D plot
        fig2 = plot_pca_3d(X_pca)
        st.pyplot(fig2)
        st.download_button("Download 3D PCA Plot (JPEG)", fig_to_jpeg(fig2), file_name="pca_3d.jpeg")

        # Cumulative variance
        fig3 = plot_variance(pca)
        st.pyplot(fig3)
        st.download_button("Download Cumulative Variance Plot (JPEG)", fig_to_jpeg(fig3),
                           file_name="cumulative_variance.jpeg")

        # Individual variance
        fig4 = plot_variance_bar(pca)
        st.pyplot(fig4)
        st.download_button("Download Variance Bar Plot (JPEG)", fig_to_jpeg(fig4), file_name="variance_bar.jpeg")

        # Loadings
        fig5 = plot_loadings(pca, features)
        st.pyplot(fig5)
        st.download_button("Download Loadings Plot (JPEG)", fig_to_jpeg(fig5), file_name="loadings_plot.jpeg")



        # Load and process data
        df = map.load_sample_data()
        df_grouped = map.group_data(df)

        # Plot and get path to JPEG
        image_path = map.plot_map(df_grouped)

        # Display image in app
        st.image(image_path, caption="ΔFMQ Global Map", use_column_width=True)

        # Optional: Show CSV download
        csv = df_grouped.to_csv(index=False).encode('utf-8')
        st.download_button("Download ΔFMQ CSV", csv, "grouped_map_data.csv", "text/csv")

