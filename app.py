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
st.set_page_config(page_title="India LST & UHI Dashboard", layout="wide")
st.title("🏙️ Land Surface Temperature & Urban Heat Island Analysis")
st.write("Upload individual Landsat Level-2 bands and a boundary shapefile to compute LST, analyze UHI intensity, and download processed geodata.")

# --- SIDEBAR: FILE UPLOADS ---
st.sidebar.header("1. Upload Satellite Bands")
st.sidebar.markdown("Upload the unzipped **Level-2** Landsat .TIF files.")
red_file = st.sidebar.file_uploader("Upload Red Band (SR_B4)", type=["tif", "tiff"])
nir_file = st.sidebar.file_uploader("Upload NIR Band (SR_B5)", type=["tif", "tiff"])
thermal_file = st.sidebar.file_uploader("Upload Thermal Band (ST_B10)", type=["tif", "tiff"])

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

def process_uploaded_raster(uploaded_file, geometries=None, return_meta=False, get_crs_only=False):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        with rasterio.open(tmp_path) as src:
            if get_crs_only:
                return src.crs
            out_image, out_transform = mask(src, geometries, crop=True)
            array = out_image[0].astype('float32')
            array[array == 0] = np.nan  # Mask background
            if return_meta:
                return array, out_transform, src.crs
            return array
    finally:
        os.remove(tmp_path)

def calculate_lst_level2(red, nir, thermal):
    np.seterr(divide='ignore', invalid='ignore')
    # NDVI
    red_sr = (red * 0.0000275) - 0.2
    nir_sr = (nir * 0.0000275) - 0.2
    ndvi = (nir_sr - red_sr) / (nir_sr + red_sr)
    
    # LST
    lst_kelvin = (thermal * 0.00341802) + 149.0
    lst_celsius = lst_kelvin - 273.15
    lst_celsius[(lst_celsius < -10) | (lst_celsius > 60)] = np.nan
    return lst_celsius, ndvi

def compute_uhi_intensity(lst_array, ndvi_array):
    """Calculates UHI intensity by comparing pixels to a rural/vegetated baseline."""
    # Define baseline using healthy vegetation (NDVI > 0.45)
    baseline_mask = (ndvi_array > 0.45) & (~np.isnan(lst_array))
    
    if np.sum(baseline_mask) > 50: # Ensure enough baseline pixels exist
        baseline_temp = np.nanmean(lst_array[baseline_mask])
    else:
        # Fallback to lowest 10th percentile if no dense vegetation exists
        baseline_temp = np.nanpercentile(lst_array, 10)
        
    uhi_intensity = lst_array - baseline_temp
    # Ignore areas cooler than baseline for UHI mapping
    uhi_intensity[uhi_intensity < 0] = 0 
    return uhi_intensity, baseline_temp

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
if red_file and nir_file and thermal_file and uploaded_shape:
    with st.spinner("Processing satellite algorithms and computing UHI intensity..."):
        try:
            # 1. Shapefile Setup
            shp_path = extract_zip(uploaded_shape)
            if not shp_path:
                st.error("Invalid shapefile inside zip.")
                st.stop()
            gdf = gpd.read_file(shp_path)
            
            # 2. CRS Alignment & Clipping
            raster_crs = process_uploaded_raster(red_file, get_crs_only=True)
            if gdf.crs != raster_crs:
                gdf = gdf.to_crs(raster_crs)
            geometries = gdf.geometry.values
            
            red_band, out_transform, out_crs = process_uploaded_raster(red_file, geometries, return_meta=True)
            nir_band = process_uploaded_raster(nir_file, geometries)
            thermal_band = process_uploaded_raster(thermal_file, geometries)
            
            # 3. Mathematical Calculations
            lst_map, ndvi_map = calculate_lst_level2(red_band, nir_band, thermal_band)
            uhi_map, baseline_temp = compute_uhi_intensity(lst_map, ndvi_map)
            
            # --- OVERALL METRICS ---
            st.success("Spatial Processing Complete!")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Average City Temp", f"{np.nanmean(lst_map):.2f} °C")
            c2.metric("Max Surface Temp", f"{np.nanmax(lst_map):.2f} °C")
            c3.metric("Rural Baseline Temp", f"{baseline_temp:.2f} °C")
            c4.metric("Max UHI Intensity (ΔT)", f"+{np.nanmax(uhi_map):.2f} °C", delta_color="inverse")
            
            # --- PRIMARY VISUALIZATIONS (LST & NDVI) ---
            st.subheader("🗺️ Base Layers: LST & Vegetation")
            fig1, ax1 = plt.subplots(1, 2, figsize=(16, 6))
            
            # LST Map (ArcGIS Style)
            lst_cmap = LinearSegmentedColormap.from_list("LST_Ramp", ["#228B22", "#FFFF00", "#FFA500", "#FF0000"])
            im1 = ax1[0].imshow(lst_map, cmap=lst_cmap, vmin=np.nanpercentile(lst_map, 2), vmax=np.nanpercentile(lst_map, 98))
            ax1[0].set_title("Land Surface Temperature (°C)")
            fig1.colorbar(im1, ax=ax1[0], label="Temp (°C)", fraction=0.046, pad=0.04)
            ax1[0].axis('off')
            
            # NDVI Map
            im2 = ax1[1].imshow(ndvi_map, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
            ax1[1].set_title("Vegetation Density Index (NDVI)")
            fig1.colorbar(im2, ax=ax1[1], label="NDVI Index", fraction=0.046, pad=0.04)
            ax1[1].axis('off')
            st.pyplot(fig1)
            
            # --- UHI & CRITICAL ANALYSIS ---
            st.markdown("---")
            st.header("📊 Urban Heat Island & Policy Analysis")
            
            # Area calculations (30m Landsat resolution)
            pixel_sqkm = (30 * 30) / 1_000_000
            total_sqkm = np.sum(~np.isnan(lst_map)) * pixel_sqkm
            severe_uhi_sqkm = np.sum(uhi_map >= 4.0) * pixel_sqkm  # Areas 4°C hotter than baseline
            
            valid_mask = ~np.isnan(lst_map) & ~np.isnan(ndvi_map)
            correlation = np.corrcoef(ndvi_map[valid_mask], lst_map[valid_mask])[0, 1]
            
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Total Analyzed Area", f"{total_sqkm:.2f} km²")
            sc2.metric("Severe Heat Stress Area (ΔT > 4°C)", f"{severe_uhi_sqkm:.2f} km²")
            sc3.metric("Vegetation-Cooling Correlation", f"{correlation:.2f}", "Negative indicates cooling effect")
            
            fig2, ax2 = plt.subplots(1, 2, figsize=(16, 6))
            
            # UHI Intensity Map
            uhi_cmap = LinearSegmentedColormap.from_list("UHI_Ramp", ["#ffffff", "#ffcc99", "#ff0000", "#660000"])
            im3 = ax2[0].imshow(uhi_map, cmap=uhi_cmap, vmin=0, vmax=np.nanpercentile(uhi_map, 98))
            ax2[0].set_title("UHI Intensity (ΔT from Baseline)")
            fig2.colorbar(im3, ax=ax2[0], label="Temperature Anomaly (°C)", fraction=0.046, pad=0.04)
            ax2[0].axis('off')
            
            # Action Target Map
            action_map = np.zeros_like(uhi_map)
            action_map[np.isnan(uhi_map)] = np.nan
            action_map[(uhi_map >= 2.0) & (uhi_map < 4.0)] = 1  # Moderate
            action_map[uhi_map >= 4.0] = 2                      # Severe
            
            cmap_action = plt.matplotlib.colors.ListedColormap(['#e0e0e0', '#ff9999', '#cc0000'])
            ax2[1].imshow(action_map, cmap=cmap_action)
            ax2[1].set_title("Vulnerability Zoning for Intervention")
            ax2[1].axis('off')
            
            # Legend for Action Map
            labels = ['Normal (ΔT < 2°C)', 'Moderate Stress (ΔT 2-4°C)', 'Severe Crisis (ΔT > 4°C)']
            colors = ['#e0e0e0', '#ff9999', '#cc0000']
            patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(3)]
            ax2[1].legend(handles=patches, loc='lower right', fontsize='small')
            
            st.pyplot(fig2)
            
            st.info(f"""
            **📋 Planner's Summary Insight:**
            * The city's rural/vegetated baseline temperature is **{baseline_temp:.1f}°C**. 
            * **{severe_uhi_sqkm:.2f} km²** of the study area operates at an extreme thermal deficit (more than 4°C hotter than surrounding nature). 
            * The strong negative correlation (**{correlation:.2f}**) mathematically proves that the neighborhoods marked in **Dark Red** on the zoning map will benefit significantly from immediate afforestation and "cool roof" mandates.
            """)
            
            # --- EXPORT DATA ---
            st.markdown("---")
            st.subheader("💾 Export GIS Layers")
            st.write("Download the resulting maps as georeferenced .TIF files for QGIS/ArcGIS.")
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                lst_tiff_bytes = create_download_tiff(lst_map, out_transform, out_crs)
                st.download_button("⬇️ Download Absolute LST Map (.tif)", data=lst_tiff_bytes, file_name="Absolute_LST_Map.tif", mime="image/tiff")
                
            with col_d2:
                uhi_tiff_bytes = create_download_tiff(uhi_map, out_transform, out_crs)
                st.download_button("⬇️ Download UHI Intensity Map (.tif)", data=uhi_tiff_bytes, file_name="UHI_Intensity_Map.tif", mime="image/tiff")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("💡 Ready. Please upload your 3 Landsat bands and your vector shapefile boundary.")
