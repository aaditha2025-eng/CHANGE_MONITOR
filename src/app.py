import streamlit as st
import os
import inference
import datetime
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Earth Observation Analytics",
    page_icon="🌍",
    layout="wide"
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("🌍 Geospatial AI Platform")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Core Philosophy:**
We are building an intelligence layer on top of satellite data, NOT just another map viewer.

**Our Pipeline:**
1. Collect Satellite Imagery
2. Run Siamese U-Net Analysis
3. Classify Semantic Change
4. Prioritize & Alert
""")

# ---------------------------------------------------------
# Main Dashboard
# ---------------------------------------------------------
st.title("AI-Powered Earth Observation Analytics")
st.markdown("Automated change detection and prioritization for disaster response, infrastructure, and agriculture.")
st.markdown("---")

st.header("Step 1: Select Analysis Domain")
selected_model = st.selectbox(
    "Choose the specific pre-trained model for your domain:",
    [
        "Disaster/Damage (xBD / xView2)",
        "Disaster/Flood (Sen1Floods11)",
        "Infrastructure/Urban (SpaceNet 7)",
        "Road Infrastructure (SpaceNet 5)",
        "Agriculture (CropHarvest)",
        "Agriculture (Radiant MLHub)"
    ]
)

st.markdown("---")
st.header("Step 2: Define Location & Upload Imagery")
st.info("Since we are uploading local files (which lack embedded GPS metadata), provide the geographic coordinates to anchor the interactive map.")

coord_col1, coord_col2 = st.columns(2)
demo_lat = coord_col1.number_input("Target Latitude", value=13.0827, format="%.4f")
demo_lon = coord_col2.number_input("Target Longitude", value=80.2707, format="%.4f")

st.markdown("---")
col1, col2 = st.columns(2)
before_file = col1.file_uploader("Upload 'Before' Image (Time 1)", type=["jpg", "jpeg", "png", "tif", "tiff"])
after_file = col2.file_uploader("Upload 'After' Image (Time 2)", type=["jpg", "jpeg", "png", "tif", "tiff"])

if before_file and after_file:
    from PIL import Image
    b_image = Image.open(before_file).convert("RGB")
    a_image = Image.open(after_file).convert("RGB")
    col1.image(b_image, caption="Historical Observation", use_container_width=True)
    col2.image(a_image, caption="Recent Observation", use_container_width=True)
    
    st.markdown("---")
    
    if st.button("🚀 Execute AI Change Detection Pipeline", type="primary", use_container_width=True):
        with st.spinner("Running deep learning semantic segmentation..."):
            os.makedirs("data/cached", exist_ok=True)
            
            # Save with original extensions to support TIFF
            b_ext = before_file.name.split('.')[-1].lower()
            a_ext = after_file.name.split('.')[-1].lower()
            b_path = f"data/cached/temp_upload_before.{b_ext}"
            a_path = f"data/cached/temp_upload_after.{a_ext}"
            
            with open(b_path, "wb") as f:
                f.write(before_file.getbuffer())
            with open(a_path, "wb") as f:
                f.write(after_file.getbuffer())
                
            # Run Inference
            res = inference.run_local_inference(b_path, a_path, domain=selected_model)
            st.session_state.inference_result = res
            st.session_state.inference_paths = (b_path, a_path)
            st.session_state.inference_domain = selected_model
            
    # --- RENDER RESULTS OUTSIDE THE BUTTON BLOCK SO THEY SURVIVE MAP RERUNS ---
    if "inference_result" in st.session_state and st.session_state.inference_result:
        res = st.session_state.inference_result
        b_path, a_path = st.session_state.inference_paths
        domain = st.session_state.inference_domain
        
        st.success("✅ Analysis Complete")
        
        st.markdown("### 🔍 Step 3: Actionable Intelligence Card")
        
        # Construct the exact output from the PDF
        priority = "🔴 HIGH PRIORITY" if res['percent_changed'] > 10 else "🟡 MEDIUM PRIORITY" if res['percent_changed'] > 3 else "🟢 LOW PRIORITY"
        recommendation = "Immediate Field Verification Required" if res['percent_changed'] > 10 else "Monitor in next satellite pass"
        
        # Intelligence Card UI
        card_col, image_col = st.columns([1, 2])
        
        with card_col:
            st.markdown(f"#### {priority}")
            st.markdown(f"**Change Detected**")
            st.markdown(f"- **Type:** {res['classification']}")
            st.markdown(f"- **Affected Area:** {res['percent_changed']:.1f}% of observed region")
            st.markdown(f"- **Confidence:** {res['confidence']}")
            st.markdown(f"- **Recommendation:** {recommendation}")
            st.markdown("---")
            st.markdown(f"**Execution Engine:** {res['model_used']}")
            
            # Generate Report Content
            report_content = f"""==================================================
EARTH OBSERVATION INTELLIGENCE REPORT
==================================================
Date Generated:  {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Domain Analyzed: {domain}

--- ANALYSIS RESULTS ---
Change Detected: {res['classification']}
Affected Area:   {res['percent_changed']:.2f}% of observed region
Confidence:      {res['confidence']}

--- RECOMMENDATION ---
Priority Level:  {priority.replace('🔴', '').replace('🟡', '').replace('🟢', '').strip()}
Action:          {recommendation}

--- TECHNICAL DETAILS ---
Execution Engine: {res['model_used']}
==================================================
"""
            
            st.download_button(
                label="📥 Download Official Report",
                data=report_content,
                file_name="EO_Intelligence_Report.txt",
                mime="text/plain",
                type="secondary"
            )
            
        with image_col:
            res_subcol1, res_subcol2 = st.columns(2)
            res_subcol1.image(res['mask_path'], caption="AI Feature Mask", use_container_width=True)
            res_subcol2.image(res['highlight_path'], caption="Static Geospatial Overlay", use_container_width=True)

        st.markdown("---")
        st.markdown("### 🗺️ Interactive Geospatial Projection")
        st.info(f"The uploaded imagery and AI change mask projected at coordinates [{demo_lat}, {demo_lon}]. Use the layer control (top right) to toggle the mask.")
        
        # Attempt Automatic GeoTIFF Coordinate Extraction
        extracted_bounds = None
        try:
            import rasterio
            from rasterio.warp import transform_bounds
            
            if b_path.lower().endswith(('.tif', '.tiff')):
                with rasterio.open(b_path) as src:
                    if src.crs:
                        # Transform bounds from native CRS to standard GPS (EPSG:4326)
                        # transform_bounds returns (west, south, east, north)
                        west, south, east, north = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
                        
                        # Folium expects bounds in [[lat_min, lon_min], [lat_max, lon_max]]
                        extracted_bounds = [[south, west], [north, east]]
                        lat = (south + north) / 2
                        lon = (west + east) / 2
        except Exception as e:
            # Silent fail for standard images or missing libraries
            pass
            
        if extracted_bounds:
            st.success(f"🌍 Geospatial metadata detected! Auto-centering map at [{lat:.4f}, {lon:.4f}]")
            bounds = extracted_bounds
            m = folium.Map(location=[lat, lon], zoom_start=15)
        else:
            # Fall back to user-defined manual center
            lat, lon = demo_lat, demo_lon
            m = folium.Map(location=[lat, lon], zoom_start=15)
            # Generic bounds (~2x2 km bounding box)
            bounds = [[lat - 0.01, lon - 0.015], [lat + 0.01, lon + 0.015]]
        
        import base64
        def get_base64_image(image_path):
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode()
            ext = image_path.split('.')[-1].lower()
            mime = "image/png" if ext == "png" else "image/jpeg"
            return f"data:{mime};base64,{encoded}"
            
        # Base64 encode the images so the browser can render them securely
        a_b64 = get_base64_image(a_path)
        mask_b64 = get_base64_image(res['transparent_mask_path'])
        
        # Add the base image
        folium.raster_layers.ImageOverlay(
            image=a_b64,
            bounds=bounds,
            opacity=1.0,
            name="Recent Satellite Imagery",
            interactive=True,
            cross_origin=False,
            zindex=1
        ).add_to(m)
        
        # Add the transparent AI mask exactly over it
        folium.raster_layers.ImageOverlay(
            image=mask_b64,
            bounds=bounds,
            opacity=0.85,
            name="🔴 AI Change Detection Mask",
            interactive=True,
            cross_origin=False,
            zindex=2
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        st_folium(m, width="1000", height=500)
