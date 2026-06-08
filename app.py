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
st.write("Upload Landsat Level-1 bands and a boundary shapefile to compute LST, analyze standardized UHI intensity, and export GIS data.")

# --- SIDEBAR: FILE UPLOADS ---
st.sidebar.header("1. Upload Satellite Bands")
st.sidebar.markdown("Upload the unzipped **Level-1** Landsat .TIF files.")
red_file = st.sidebar.file_uploader("Upload Red Band (Band 4)", type=["tif", "tiff"])
nir_file = st.sidebar.file_uploader("Upload NIR Band (Band 5)", type=["tif", "tiff"])
thermal_file = st.sidebar.file_uploader("Upload Thermal Band (Band 10)", type=["tif", "tiff"])

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
            array[array == 0] = np.nan  # Mask background clipping artifacts
            if return_meta:
                return array, out_transform, src.crs
            return array
    finally:
        os.remove(tmp_path)

def calculate_lst_level1(red, nir, thermal):
    """Calculates LST from raw Level-1 Digital Numbers."""
    np.seterr(divide='ignore', invalid='ignore')
    
    # 1. TOA Spectral Radiance
    radiance = (thermal * 0.0003342) + 0.1
    
    # 2. At-Satellite Brightness Temperature (Celsius)
    K1 = 774.8853
    K2 = 1321.0789
    bt_celsius = (K2 / np.log((K1 / radiance) + 1)) - 273.15
    
    # 3. NDVI
    ndvi = (nir - red) / (nir + red)
    
    # 4. Proportion of Vegetation (Pv)
    ndvi_min, ndvi_max = 0.2, 0.5
    pv = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min)) ** 2
    pv = np.clip(pv, 0, 1)
    
    # 5. Land Surface Emissivity (Epsilon)
    emissivity = 0.004 * pv + 0.986
    
    # 6. Final Land Surface Temperature
    lam = 10.895
    rho = 14388
    lst = bt_celsius / (1 + ((lam * bt_celsius / rho) * np.log(emissivity)))
    
    # Filter extreme atmospheric/clipping anomalies
    lst[(lst < -10) | (lst > 65)] = np.nan
    return lst, ndvi

def compute_uhi_standardized(lst_array):
    """Calculates the UHI Index using standard deviation normalization (Z-score)."""
    mean_lst = np.nanmean(lst_array)
    std_lst = np.nanstd(lst_array)
    
    # UHI = (LST - Mean) / StdDev
    uhi_index = (lst_array - mean_lst) / std_lst
    
    return uhi_index, mean_lst, std_lst

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
    with st.spinner("Processing Level-1 satellite algorithms and standardizing UHI..."):
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
            lst_map, ndvi_map = calculate_lst_level1(red_band, nir_band, thermal_band)
            uhi_map, mean_lst, std_lst = compute_uhi_standardized(lst_map)
            
            # --- OVERALL METRICS ---
            st.success("Spatial Processing Complete!")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Study Area Mean (μ)", f"{mean_lst:.2f} °C")
            c2.metric("Standard Deviation (σ)", f"{std_lst:.2f} °C")
            c3.metric("Max Urban Temp", f"{np.nanmax(lst_map):.2f} °C")
            c4.metric("Max UHI Index Score", f"+{np.nanmax(uhi_map):.2f} σ", delta_color="inverse")
            
            # --- PRIMARY VISUALIZATIONS (LST & NDVI) ---
            st.subheader("🗺️ Base Layers: LST & Vegetation")
            fig1, ax1 = plt.subplots(1, 2, figsize=(16, 6))
            
            # LST Map
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
            
            # --- STANDARDIZED UHI ANALYSIS ---
            st.markdown("---")
            st.header("📊 Standardized Urban Heat Island Analysis")
            
            # Area calculations (30m Landsat resolution)
            pixel_sqkm = (30 * 30) / 1_000_000
            total_sqkm = np.sum(~np.isnan(lst_map)) * pixel_sqkm
            
            # Calculate zones based on Z-score (Standard Deviations)
            moderate_uhi_sqkm = np.sum((uhi_map >= 1.0) & (uhi_map < 2.0)) * pixel_sqkm
            severe_uhi_sqkm = np.sum(uhi_map >= 2.0) * pixel_sqkm
            
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Total Analyzed Area", f"{total_sqkm:.2f} km²")
            sc2.metric("Moderate UHI Area (1σ to 2σ)", f"{moderate_uhi_sqkm:.2f} km²")
            sc3.metric("Severe UHI Crisis Area (> 2σ)", f"{severe_uhi_sqkm:.2f} km²", "Requires Intervention")
            
            fig2, ax2 = plt.subplots(1, 2, figsize=(16, 6))
            
            # UHI Standardized Index Map (Blue to Red Diverging)
            im3 = ax2[0].imshow(uhi_map, cmap='RdYlBu_r', vmin=-3, vmax=3)
            ax2[0].set_title("UHI Index (Standard Deviations from Mean)")
            fig2.colorbar(im3, ax=ax2[0], label="UHI Score (Z-Value)", fraction=0.046, pad=0.04)
            ax2[0].axis('off')
            
            # Action Target Map based on Standard Deviations
            action_map = np.zeros_like(uhi_map)
            action_map[np.isnan(uhi_map)] = np.nan
            action_map[(uhi_map >= 1.0) & (uhi_map < 2.0)] = 1  # Moderate (1 to 2 StdDev)
            action_map[uhi_map >= 2.0] = 2                      # Severe (> 2 StdDev)
            
            cmap_action = plt.matplotlib.colors.ListedColormap(['#e0e0e0', '#ff9999', '#cc0000'])
            ax2[1].imshow(action_map, cmap=cmap_action)
            ax2[1].set_title("Vulnerability Zoning (Based on σ)")
            ax2[1].axis('off')
            
            # Legend for Action Map
            labels = ['Average/Cool (UHI < 1)', 'Moderate Heat (UHI 1 to 2)', 'Severe Heat (UHI > 2)']
            colors = ['#e0e0e0', '#ff9999', '#cc0000']
            patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(3)]
            ax2[1].legend(handles=patches, loc='lower right', fontsize='small')
            
            st.pyplot(fig2)
            
            st.info(f"""
            **📋 Planner's Standardization Insight:**
            * The UHI Index is now calculated as a **Z-Score** ($UHI = \\frac{{LST - \\mu}}{{\\sigma}}$). 
            * A score of `0` represents the exact average temperature of the study area ({mean_lst:.2f}°C).
            * Red zones on the vulnerability map indicate areas that are **more than 2 standard deviations hotter** than the rest of the city, objectively highlighting the most statistically critical heat islands.
            """)
            
            # --- EXPORT DATA ---
            st.markdown("---")
            st.subheader("💾 Export GIS Layers")
            st.write("Download the maps as georeferenced .TIF files for QGIS/ArcGIS.")
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                lst_tiff_bytes = create_download_tiff(lst_map, out_transform, out_crs)
                st.download_button("⬇️ Download Absolute LST Map (.tif)", data=lst_tiff_bytes, file_name="Level1_LST_Map.tif", mime="image/tiff")
                
            with col_d2:
                uhi_tiff_bytes = create_download_tiff(uhi_map, out_transform, out_crs)
                st.download_button("⬇️ Download UHI Standardized Index Map (.tif)", data=uhi_tiff_bytes, file_name="Standardized_UHI_Map.tif", mime="image/tiff")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("💡 Ready. Please upload your 3 Level-1 Landsat bands and your vector shapefile boundary.")
