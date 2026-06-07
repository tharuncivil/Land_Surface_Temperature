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

# Page configuration
st.set_page_config(page_title="India LST Mapping Dashboard", layout="wide")
st.title("🏙️ Land Surface Temperature (LST) Analysis Dashboard")
st.write("Upload individual Landsat Level-2 bands and a study area boundary to compute and download LST.")

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
            array[array == 0] = np.nan
            
            if return_meta:
                return array, out_transform, src.crs
            return array
    finally:
        os.remove(tmp_path)

def calculate_lst_level2(red, nir, thermal):
    np.seterr(divide='ignore', invalid='ignore')
    
    # 1. NDVI 
    red_sr = (red * 0.0000275) - 0.2
    nir_sr = (nir * 0.0000275) - 0.2
    ndvi = (nir_sr - red_sr) / (nir_sr + red_sr)
    
    # 2. LST (Kelvin to Celsius)
    lst_kelvin = (thermal * 0.00341802) + 149.0
    lst_celsius = lst_kelvin - 273.15
    
    lst_celsius[(lst_celsius < -10) | (lst_celsius > 60)] = np.nan
    return lst_celsius, ndvi

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
    with st.spinner("Clipping Level-2 imagery to boundary and scaling data..."):
        try:
            # 1. Handle shapefile
            shp_path = extract_zip(uploaded_shape)
            if not shp_path:
                st.error("Could not find a valid .shp file inside the zipped folder.")
                st.stop()
                
            gdf = gpd.read_file(shp_path)
            
            # 2. Extract CRS to align datasets
            raster_crs = process_uploaded_raster(red_file, get_crs_only=True)
            if gdf.crs != raster_crs:
                gdf = gdf.to_crs(raster_crs)
                
            geometries = gdf.geometry.values
            
            # 3. Clip imagery and capture spatial metadata safely
            red_band, out_transform, out_crs = process_uploaded_raster(red_file, geometries, return_meta=True)
            nir_band = process_uploaded_raster(nir_file, geometries)
            thermal_band = process_uploaded_raster(thermal_file, geometries)
            
            # 4. Run Level-2 Algorithm
            lst_map, ndvi_map = calculate_lst_level2(red_band, nir_band, thermal_band)
            
            # --- DASHBOARD METRICS DISPLAY ---
            st.success("Spatial Processing Complete!")
            col1, col2, col3 = st.columns(3)
            
            mean_lst = np.nanmean(lst_map)
            max_lst = np.nanmax(lst_map)
            min_lst = np.nanmin(lst_map)
            
            col1.metric("Average Surface Temp", f"{mean_lst:.2f} °C")
            col2.metric("Max Urban Temperature", f"{max_lst:.2f} °C")
            col3.metric("Min Cooler Zone Temp", f"{min_lst:.2f} °C")
            
            # --- VISUALIZATION MAPS ---
            st.subheader("🗺️ Spatial Analytics Visualization")
            fig, ax = plt.subplots(1, 2, figsize=(16, 7))
            
            # 1. Custom ArcGIS-style colormap for LST
            lst_colors = ["#228B22", "#FFFF00", "#FFA500", "#FF0000"]
            lst_cmap = LinearSegmentedColormap.from_list("LST_Ramp", lst_colors)
            lst_vmin, lst_vmax = np.nanpercentile(lst_map, 2), np.nanpercentile(lst_map, 98)
            
            im1 = ax[0].imshow(lst_map, cmap=lst_cmap, vmin=lst_vmin, vmax=lst_vmax)
            ax[0].set_title("Land Surface Temperature (°C)")
            fig.colorbar(im1, ax=ax[0], label="Temp (°C)", fraction=0.046, pad=0.04)
            ax[0].axis('off')
            
            # 2. NDVI Map
            im2 = ax[1].imshow(ndvi_map, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
            ax[1].set_title("Vegetation Density Index (NDVI)")
            fig.colorbar(im2, ax=ax[1], label="NDVI Index", fraction=0.046, pad=0.04)
            ax[1].axis('off')
            
            st.pyplot(fig)
            
            # --- CRITICAL ANALYSIS FOR URBAN PLANNERS ---
            st.markdown("---")
            st.header("🏙️ Urban Planning & Policy Insights")
            
            # Spatial calculations (Landsat resolution is 30m x 30m = 900 sqm per pixel)
            pixel_area_sqm = 30 * 30
            total_valid_pixels = np.sum(~np.isnan(lst_map))
            total_area_sqkm = (total_valid_pixels * pixel_area_sqm) / 1_000_000
            
            # UHI Thresholds (Standard Deviation Approach)
            std_lst = np.nanstd(lst_map)
            uhi_threshold = mean_lst + std_lst
            severe_uhi_threshold = mean_lst + (1.5 * std_lst)
            
            uhi_pixels = np.sum(lst_map >= uhi_threshold)
            uhi_area_sqkm = (uhi_pixels * pixel_area_sqm) / 1_000_000
            uhi_percentage = (uhi_pixels / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0
            
            # Statistical Correlation between Heat and Vegetation
            valid_mask = ~np.isnan(lst_map) & ~np.isnan(ndvi_map)
            valid_lst = lst_map[valid_mask]
            valid_ndvi = ndvi_map[valid_mask]
            correlation = np.corrcoef(valid_ndvi, valid_lst)[0, 1]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Study Area Analyzed", f"{total_area_sqkm:.2f} km²")
            c2.metric("Critical UHI Action Area", f"{uhi_area_sqkm:.2f} km²", f"{uhi_percentage:.1f}% of city", delta_color="inverse")
            c3.metric("NDVI vs Heat Correlation", f"{correlation:.2f}", "Negative = strong cooling effect")
            
            st.subheader("Actionable Target Zones & Distribution")
            fig2, ax2 = plt.subplots(1, 2, figsize=(16, 6))
            
            # Actionable Binary Map
            action_map = np.zeros_like(lst_map)
            action_map[np.isnan(lst_map)] = np.nan
            action_map[lst_map >= uhi_threshold] = 1
            action_map[lst_map >= severe_uhi_threshold] = 2
            
            cmap_action = plt.matplotlib.colors.ListedColormap(['#e0e0e0', '#ff9999', '#cc0000'])
            ax2[0].imshow(action_map, cmap=cmap_action)
            ax2[0].set_title("Urban Heat Island (UHI) Target Zones")
            ax2[0].axis('off')
            
            # Custom Legend for Action Map
            labels = ['Normal/Cool', f'High Heat (>{uhi_threshold:.1f}°C)', f'Severe Heat (>{severe_uhi_threshold:.1f}°C)']
            colors = ['#e0e0e0', '#ff9999', '#cc0000']
            patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(3)]
            ax2[0].legend(handles=patches, loc='lower right', fontsize='small')
            
            # Temperature Histogram
            ax2[1].hist(valid_lst, bins=50, color='orange', edgecolor='black', alpha=0.7)
            ax2[1].axvline(mean_lst, color='blue', linestyle='dashed', linewidth=2, label=f'Mean: {mean_lst:.1f}°C')
            ax2[1].axvline(uhi_threshold, color='red', linestyle='dashed', linewidth=2, label=f'UHI Alert: {uhi_threshold:.1f}°C')
            ax2[1].set_title("Surface Temperature Distribution")
            ax2[1].set_xlabel("Temperature (°C)")
            ax2[1].set_ylabel("Pixel Count")
            ax2[1].legend()
            
            st.pyplot(fig2)
            
            st.info("""
            **📋 Planner's Summary Note:**
            * **Severe Heat Targeting:** Red zones on the target map represent areas operating above 1.5 standard deviations from the city average. These micro-locations require immediate policy interventions like cool-roof mandates or reflective paving.
            * **Vegetation Link:** The correlation metric shows how strongly vegetation mitigates heat. A negative correlation approaching -1.0 proves that targeted green canopy expansion in the red zones will yield highly predictable surface cooling.
            """)
            
            # --- GEO-DATA EXPORT ---
            st.markdown("---")
            st.subheader("💾 Export Processed Data")
            tiff_bytes = create_download_tiff(lst_map, out_transform, out_crs)
            
            st.download_button(
                label="⬇️ Download LST GeoTIFF",
                data=tiff_bytes,
                file_name="Processed_LST_Map.tif",
                mime="image/tiff"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {e}")