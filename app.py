import io
import zipfile
from datetime import date
import pandas as pd
import streamlit as st
from detector import process_single_image

st.set_page_config(page_title="Rice Stomatal Density Calculator", layout="wide")
st.title("Calculate rice stomatal density from ~400x images using YOLOv11 machine learning model)")
st.caption("Upload Dino-lite images of rice leaves (abaxial or adaxial) to calculate density!")

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
    value=0.5,
    step=0.5,
    help="Excludes stomates touching the image boarder"
)

# --- Main Upload & Processing ---
uploaded_files = st.file_uploader(
    "Upload Dino-lite microscope images",
    type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
    accept_multiple_files=True
)

if uploaded_files:
    if not api_key:
        st.warning("Please enter your Roboflow API Key in the sidebar to run inference.")
        st.stop()

    if st.button(f"Analyse {len(uploaded_files)} Image(s)", type="primary"):
        results = []
        progress = st.progress(0)

        for idx, file in enumerate(uploaded_files):
            try:
                res = process_single_image(
                    api_key=api_key,
                    image_bytes=file.getvalue(),
                    filename=file.name,
                    image_width_mm=image_width_mm,
                    confidence=confidence,
                    edge_margin=edge_margin
                )
                results.append(res)
            except Exception as error:
                st.error(f"Skipping {file.name}: {error}")

            progress.progress((idx + 1) / len(uploaded_files))

        # Store results in session_state so they stay visible when clicking download buttons
        st.session_state["results"] = results

# --- Display Results & Downloads ---
if "results" in st.session_state and st.session_state["results"]:
    results = st.session_state["results"]

    # Build summary DataFrame
    df = pd.DataFrame([
        {
            "image": r["image"],
            "stomate_count": r["stomate_count"],
            "crop_area_mm2": r["crop_area_mm2"],
            "stomatal_density_per_mm2": r["stomatal_density_per_mm2"]
        }
        for r in results
    ])

    st.subheader("Results Summary")
    st.dataframe(df, use_container_width=True)

    # Create ZIP archive of all labelled images in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            zf.writestr(r["labelled_filename"], r["annotated_bytes"])

    today_str = date.today().strftime("%d_%m_%y")
    col_csv, col_zip = st.columns(2)
    col_csv.download_button(
        label="Download Results CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{today_str}_stomatal_density.csv",
        mime="text/csv"
    )
    col_zip.download_button(
        label="Download Labelled Images (ZIP)",
        data=zip_buffer.getvalue(),
        file_name=f"{today_str}_labelled_images.zip",
        mime="application/zip"
    )

    # Visual verification gallery
    st.subheader("Labelled Image Inspection")
    cols = st.columns(3)
    for idx, r in enumerate(results):
        with cols[idx % 3]:
            st.image(
                r["annotated_image"],
                caption=f"{r['image']} | Count: {r['stomate_count']} ({r['stomatal_density_per_mm2']} / mm²)",
                use_container_width=True
            )
