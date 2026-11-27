import io
import zipfile
from datetime import date
import pandas as pd
import streamlit as st
from detector import process_single_image

st.set_page_config(page_title="Rice Stomatal Density Calculator", layout="wide")
st.title("Rice Stomatal Density Calculator")
st.caption("Calculate rice stomatal density from ~400x images using YOLOv11 machine learning model")

# --- Sidebar Settings ---
st.sidebar.header("Configuration")

api_key = st.sidebar.text_input(
    "Roboflow API Key",
    type="password",
    help="Found in Roboflow Settings > API Keys"
)

image_width_mm = st.sidebar.number_input(
    "Original Image Width (mm)",
    help="Measured from a calibration slide/micrometer at the same magnification as your images"
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.00,
    max_value=1.00,
    value=0.15,
    step=0.05,
    help="Lower = less stringent (risk of false positives); Higher = stricter (risk of false negatives)"
)

edge_margin = st.sidebar.number_input(
    "Edge Exclusion Margin",
    min_value=0.0,
    value=1.0,
    step=0.5,
    help="Excludes stomates touching the image boarder"
)

# --- Main Upload & Processing ---
uploaded_files = st.file_uploader(
    "Upload Dino-lite microscope images",
    type=["bmp"],
    accept_multiple_files=True
)

# process uploaded images
if uploaded_files and api_key:
    if st.button(f"Analyse {len(uploaded_files)} Image(s)"):
        results = [
            process_single_image(
                api_key=api_key,
                image_bytes=file.getvalue(),
                filename=file.name,
                image_width_mm=image_width_mm,
                confidence=confidence,
                edge_margin=edge_margin
            )
            for file in uploaded_files
        ]

# --- Display Results & Downloads ---
# build the summary table
if "results" in locals() and results:
    df = pd.DataFrame([
        {
            "image": r["image"],
            "stomate_count": r["stomate_count"],
            "stomatal_density_per_mm2": r["stomatal_density_per_mm2"]
        }
        for r in results
    ])

    # display results
    st.subheader("Results Summary")
    st.dataframe(df, use_container_width=True)

    # csv download button
    st.download_button("Download Results CSV", df.to_csv(index=False), "stomatal_density.csv")

    # Visual verification gallery
    st.subheader("Labelled Image Inspection")
    cols = st.columns(8)
    for idx, r in enumerate(results):
        with cols[idx % 8]:
            st.image(
                r["annotated_image"],
                caption=f"{r['image']} - Count: {r['stomate_count']}"
            )
