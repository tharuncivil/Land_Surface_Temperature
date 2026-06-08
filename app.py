
import streamlit as st
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import zipfile
import os
import tempfile

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Drone UAV Urban Surface Analytics", layout="wide")
st.title("🚁 High-Resolution Drone Urban Surface Analyzer")
st.write("Upload a UAV RGB GeoTIFF and study area boundary to map micro-level vegetation (VARI) and thermal risk potential.")

# --- SIDEBAR: FILE UPLOADS ---
st.sidebar.header("1. Upload Drone Imagery")
st.sidebar.markdown("Upload your converted **RGB GeoTIFF** (.tif).")
drone_file = st.sidebar.file_uploader("Upload Drone Orthomosaic", type=["tif", "tiff"])

st.sidebar.header("2. Upload Study Area")
uploaded_shape = st.sidebar.file_uploader("Upload Boundary (Zipped Shapefile)", type=["zip"])

# --- HELPER FUNCTIONS ---
def extract_zip(uploaded_zip):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(".shp"):
                return os.path.join(root, file)
    return None

def process_drone_raster(uploaded_file, geometries=None, return_meta=False):
    """Saves upload to disk, masks it, and extracts RGB bands."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
        
    try:
        with rasterio.open(tmp_path) as src:
            # Drones can have huge resolutions. Clipping early is essential.
            out_image, out_transform = mask(src, geometries, crop=True)
            
            # Extract R, G, B (assuming standard band order 1, 2, 3)
            red = out_image[0].astype('float32')
            green = out_image[1].astype('float32')
            blue = out_image[2].astype('float32')
            
            # Mask background artifacts from clipping
            red[red == 0] = np.nan
            green[green == 0] = np.nan
            blue[blue == 0] = np.nan
            
            if return_meta:
                return red, green, blue, out_transform, src.crs
            return red, green, blue
    finally:
        os.remove(tmp_path)

def calculate_vari_and_risk(red, green, blue):
    """Calculates VARI and derives an Impervious Thermal Risk map."""
    np.seterr(divide='ignore', invalid='ignore')
    
    # 1. Calculate VARI (Add small epsilon to avoid divide by zero)
    epsilon = 1e-5
    vari = (green - red) / (green + red - blue + epsilon)
    
    # Filter extreme optical anomalies (e.g., pure white roofs causing weird math)
    vari[(vari < -1) | (vari > 1)] = np.nan
    
    # 2. Calculate Thermal Risk Potential (Inverse of Vegetation)
    # Scale VARI from -1 (Concrete/Water) to +1 (Dense Trees) into a 0 to 100 Risk Scale.
    # We invert it so Low VARI = High Risk (100), High VARI = Low Risk (0)
    risk_map = ((1 - vari) / 2) * 100
    
    # Water usually has very negative VARI but isn't a heat risk. 
    # In a full model we'd use NDWI to mask water, but for simple RGB we cap the risk.
    risk_map = np.clip(risk_map, 0, 100)
    
    return vari, risk_map

def create_download_tiff(array, transform, crs):
    profile = {
        'driver': 'GTiff',
        'height': array.shape[0],
        'width': array.shape[1],
        'count': 1,
        'dtype': str(array.dtype),
        'crs': crs,
        'transform': transform,
        'nodata': np.nan
    }
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(array, 1)
        return memfile.read()

# --- MAIN DASHBOARD LOGIC ---
if drone_file and uploaded_shape:
    with st.spinner("Clipping high-resolution drone imagery and calculating micro-analytics..."):
        try:
            # 1. Shapefile Setup
            shp_path = extract_zip(uploaded_shape)
            if not shp_path:
                st.error("Invalid shapefile inside zip.")
                st.stop()
            gdf = gpd.read_file(shp_path)
            
            # 2. Process Raster & Align CRS
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_crs:
                tmp_crs.write(drone_file.getbuffer())
                with rasterio.open(tmp_crs.name) as src:
                    raster_crs = src.crs
            os.remove(tmp_crs.name)
            
            if gdf.crs != raster_crs:
                gdf = gdf.to_crs(raster_crs)
                
            geometries = gdf.geometry.values
            
            # 3. Extract Bands & Calculate
            red, green, blue, out_transform, out_crs = process_drone_raster(drone_file, geometries, return_meta=True)
            vari_map, risk_map = calculate_vari_and_risk(red, green, blue)
            
            # --- OVERALL METRICS ---
            st.success("High-Resolution Spatial Processing Complete!")
            
            # Simple thresholding for metrics: VARI > 0.1 is generally healthy vegetation
            total_pixels = np.sum(~np.isnan(vari_map))
            veg_pixels = np.sum(vari_map > 0.1)
            high_risk_pixels = np.sum(risk_map > 75)
            
            veg_percent = (veg_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            risk_percent = (high_risk_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Green Canopy Footprint", f"{veg_percent:.1f}%", "Based on VARI > 0.1")
            c2.metric("Critical Concrete Risk Area", f"{risk_percent:.1f}%", "High potential for UHI retention", delta_color="inverse")
            c3.metric("Average Risk Score", f"{np.nanmean(risk_map):.1f} / 100")
            
            # --- VISUALIZATIONS ---
            st.markdown("---")
            st.subheader("🗺️ Micro-Level Action Maps")
            fig, ax = plt.subplots(1, 2, figsize=(16, 6))
            
            # VARI Vegetation Map
            im1 = ax[0].imshow(vari_map, cmap='YlGn', vmin=-0.2, vmax=0.5)
            ax[0].set_title("Visible Vegetation Canopy (VARI)")
            fig.colorbar(im1, ax=ax[0], label="VARI Index", fraction=0.046, pad=0.04)
            ax[0].axis('off')
            
            # Thermal Risk Potential Map
            risk_cmap = LinearSegmentedColormap.from_list("Risk", ["#228B22", "#e0e0e0", "#FFA500", "#FF0000"])
            im2 = ax[1].imshow(risk_map, cmap=risk_cmap, vmin=0, vmax=100)
            ax[1].set_title("Impervious Surface Heat Risk (0-100)")
            fig.colorbar(im2, ax[1], label="Heat Retention Risk", fraction=0.046, pad=0.04)
            ax[1].axis('off')
            
            st.pyplot(fig)
            
            st.info("""
            **📋 Planner's Micro-Climate Insight:**
            * **VARI Map:** Because drone data captures extreme detail, you can use the Green map to identify exact street trees that are unhealthy or missing.
            * **Heat Risk Map:** The Red zones highlight dense, unshaded impervious surfaces (like large factory roofs or wide asphalt intersections). These exact micro-locations are prime candidates for high-albedo paint or targeted tree planting.
            """)
            
            # --- EXPORT DATA ---
            st.markdown("---")
            st.subheader("💾 Export High-Res GIS Layers")
            st.write("Download the resulting maps as georeferenced .TIF files for QGIS/ArcGIS.")
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                vari_tiff_bytes = create_download_tiff(vari_map, out_transform, out_crs)
                st.download_button("⬇️ Download VARI Canopy Map (.tif)", data=vari_tiff_bytes, file_name="Drone_VARI_Vegetation.tif", mime="image/tiff")
                
            with col_d2:
                risk_tiff_bytes = create_download_tiff(risk_map, out_transform, out_crs)
                st.download_button("⬇️ Download Heat Risk Map (.tif)", data=risk_tiff_bytes, file_name="Drone_Heat_Risk.tif", mime="image/tiff")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("Ensure your drone file is a true GeoTIFF (.tif) and not simply renamed. If the file is extremely large (>500MB), Streamlit memory may run out.")
else:
    st.info("💡 Ready. Please upload your drone RGB GeoTIFF and your vector shapefile boundary.")
