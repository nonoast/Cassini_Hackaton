#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solar Radiation and Terrain Elevation Analysis Script

This script loads and analyzes solar radiation data along with terrain elevation
for specific geographic locations. It extracts time series data at given coordinates
and visualizes the relationship between solar radiation and terrain features.

Created for Health and Climate Hackathon MVP
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import earthkit.data as ekd
import rioxarray
import xarray as xr
from datetime import datetime, timedelta


def load_radiation_data(filepath):
    """
    Load solar radiation data using earthkit.data
    
    Parameters:
    -----------
    filepath : str
        Path to the radiation data file (GRIB or NetCDF)
        
    Returns:
    --------
    xarray.Dataset or equivalent earthkit data object
    """
    print(f"Loading radiation data from {filepath}...")
    
    # Determine file type from extension
    file_ext = os.path.splitext(filepath)[1].lower()
    
    # Use pyogrio engine with Arrow for better performance
    if file_ext in ['.grib', '.grb', '.grib2']:
        # Load GRIB data with earthkit
        radiation_data = ekd.from_source("file", filepath)
    elif file_ext in ['.nc', '.netcdf']:
        # For NetCDF files, use xarray directly which earthkit supports
        radiation_data = ekd.from_source("file", filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    return radiation_data


def load_dem_data(filepath):
    """
    Load Digital Elevation Model (DEM) data using rioxarray
    
    Parameters:
    -----------
    filepath : str
        Path to the DEM GeoTIFF file
        
    Returns:
    --------
    xarray.DataArray with elevation data
    """
    print(f"Loading DEM data from {filepath}...")
    
    # Load DEM with rioxarray for better CRS handling
    # Use chunks for large files to enable dask-based processing
    dem_data = rioxarray.open_rasterio(filepath, chunks={'x': 1000, 'y': 1000})
    
    # If DEM has multiple bands, use the first one
    if 'band' in dem_data.dims and dem_data.sizes['band'] > 1:
        dem_data = dem_data.sel(band=1)
    
    # Remove the band dimension if it exists
    if 'band' in dem_data.dims:
        dem_data = dem_data.squeeze('band')
    
    return dem_data


def extract_radiation_time_series(radiation_data, lat, lon):
    """
    Extract radiation time series at a specific location
    
    Parameters:
    -----------
    radiation_data : earthkit data object or xarray.Dataset
        The loaded radiation data
    lat, lon : float
        Latitude and longitude coordinates
        
    Returns:
    --------
    pandas.DataFrame with time series data
    """
    print(f"Extracting radiation time series at coordinates: {lat}, {lon}")
    
    # Handle different data types (earthkit FieldList vs xarray)
    if hasattr(radiation_data, 'to_xarray'):
        # Convert earthkit data to xarray for consistent processing
        data_xr = radiation_data.to_xarray()
    else:
        data_xr = radiation_data
    
    # Extract time series at the nearest grid point
    # Note: Method depends on the structure of your specific radiation data
    try:
        # First attempt - if data has explicit lat/lon coordinates
        time_series = data_xr.sel(latitude=lat, longitude=lon, method='nearest')
    except (KeyError, ValueError):
        # Alternative approach if coordinate names are different
        lat_dim = next((dim for dim in data_xr.dims if dim in ['lat', 'latitude']), None)
        lon_dim = next((dim for dim in data_xr.dims if dim in ['lon', 'longitude']), None)
        
        if lat_dim and lon_dim:
            time_series = data_xr.sel({lat_dim: lat, lon_dim: lon}, method='nearest')
        else:
            raise ValueError("Could not identify latitude/longitude dimensions in the data")
    
    # Convert to pandas DataFrame for easier time series handling
    if hasattr(time_series, 'to_dataframe'):
        df = time_series.to_dataframe()
        
        # Reset index to have time as a column if it's in the index
        if 'time' in df.index.names:
            df = df.reset_index()
            
        return df
    else:
        raise ValueError("Could not convert extracted data to DataFrame")


def extract_elevation(dem_data, lat, lon):
    """
    Extract elevation at a specific location from DEM
    
    Parameters:
    -----------
    dem_data : xarray.DataArray
        The loaded DEM data
    lat, lon : float
        Latitude and longitude coordinates
        
    Returns:
    --------
    float : Elevation value at the specified coordinates
    """
    print(f"Extracting elevation at coordinates: {lat}, {lon}")
    
    # Extract the elevation value at the given coordinates
    # First ensure CRS is correct for the extraction
    try:
        # Get elevation at point
        point_value = dem_data.sel(x=lon, y=lat, method='nearest')
        elevation = float(point_value.values)
        return elevation
    except Exception as e:
        print(f"Error extracting elevation: {e}")
        # If there's an issue with direct extraction, try the alternative method
        try:
            # Alternative method using rio accessor
            elevation = float(dem_data.rio.sample([(lon, lat)])[0])
            return elevation
        except Exception as e2:
            print(f"Error with alternative extraction method: {e2}")
            return None


def plot_radiation_time_series(df, radiation_var, elevation, lat, lon, output_path=None):
    """
    Plot radiation time series with elevation information
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the time series data
    radiation_var : str
        Name of the radiation variable to plot
    elevation : float
        Elevation at the specified coordinates
    lat, lon : float
        Latitude and longitude coordinates
    output_path : str, optional
        Path to save the plot
    """
    print("Creating time series plot...")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Determine if we're plotting hourly or daily data
    time_col = 'time' if 'time' in df.columns else df.index.name
    
    if time_col:
        # Ensure we have a datetime column/index
        if time_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                try:
                    df[time_col] = pd.to_datetime(df[time_col])
                except:
                    print("Warning: Could not convert time column to datetime")
            x_values = df[time_col]
        else:
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    print("Warning: Could not convert index to datetime")
            x_values = df.index
            
        # Plot the radiation data
        if radiation_var in df.columns:
            ax.plot(x_values, df[radiation_var], 'b-', linewidth=2)
            
            # Format x-axis based on data frequency
            time_range = max(x_values) - min(x_values)
            if time_range < timedelta(days=3):
                # For hourly data
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.xticks(rotation=45)
            else:
                # For daily data
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
            
            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Set labels and title
            ax.set_xlabel('Time')
            ax.set_ylabel(f'Solar Radiation ({get_units_for_var(radiation_var)})')
            ax.set_title(f'Solar Radiation at Lat: {lat:.4f}, Lon: {lon:.4f}\nElevation: {elevation:.1f} meters')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save if output path is provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {output_path}")
            
            plt.show()
        else:
            print(f"Warning: Radiation variable '{radiation_var}' not found in data")
    else:
        print("Warning: No time dimension found in data")


def get_units_for_var(var_name):
    """
    Return appropriate units for the given variable name
    
    Parameters:
    -----------
    var_name : str
        Name of the variable
        
    Returns:
    --------
    str : Units for the variable
    """
    # Dictionary mapping variable names to their units
    units_dict = {
        'ghi': 'W/m²',
        'dni': 'W/m²',
        'dhi': 'W/m²',
        'GHI': 'W/m²',
        'DNI': 'W/m²',
        'DHI': 'W/m²',
        'ssrd': 'J/m²',  # ERA5 surface solar radiation downwards
        'fdir': 'J/m²',  # ERA5 direct solar radiation
        'ssrdc': 'J/m²', # ERA5 surface solar radiation downwards clear-sky
        'default': 'Unknown units'
    }
    
    return units_dict.get(var_name, units_dict['default'])


def calculate_multi_day_averages(df, radiation_var, window_days=7):
    """
    Calculate moving averages for radiation data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the time series data
    radiation_var : str
        Name of the radiation variable to average
    window_days : int
        Size of the moving window in days
        
    Returns:
    --------
    pandas.DataFrame with original and averaged data
    """
    print(f"Calculating {window_days}-day moving averages...")
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Ensure we have a datetime index
    time_col = 'time' if 'time' in result_df.columns else None
    if time_col:
        result_df = result_df.set_index(time_col)
    
    # Calculate the moving average
    if radiation_var in result_df.columns:
        # Convert window days to appropriate frequency based on data
        # Determine the frequency of the data
        try:
            freq = pd.infer_freq(result_df.index)
            if freq and 'H' in freq:  # Hourly data
                window = window_days * 24  # Convert days to hours
            else:
                window = window_days  # Assume daily data
                
            # Calculate moving average
            result_df[f'{radiation_var}_{window_days}day_avg'] = (
                result_df[radiation_var].rolling(window=window, min_periods=1).mean()
            )
        except Exception as e:
            print(f"Error calculating moving average: {e}")
    
    # Reset index if we set it earlier
    if time_col:
        result_df = result_df.reset_index()
        
    return result_df


def analyze_terrain_shading_potential(dem_data, lat, lon, radius_km=5):
    """
    Analyze potential for terrain shading at the given location
    
    Parameters:
    -----------
    dem_data : xarray.DataArray
        The loaded DEM data
    lat, lon : float
        Latitude and longitude coordinates
    radius_km : float
        Radius around the point to analyze (in kilometers)
        
    Returns:
    --------
    dict : Dictionary with shading analysis results
    """
    print(f"Analyzing terrain shading potential within {radius_km} km radius...")
    
    # Convert radius from km to degrees (approximate)
    # 1 degree is roughly 111 km at the equator
    radius_deg = radius_km / 111.0
    
    try:
        # Extract a subset of the DEM around our point
        dem_subset = dem_data.sel(
            x=slice(lon - radius_deg, lon + radius_deg),
            y=slice(lat + radius_deg, lat - radius_deg)
        )
        
        # Get elevation at our point
        point_elev = extract_elevation(dem_data, lat, lon)
        
        if point_elev is None:
            return {"error": "Could not extract elevation at the specified point"}
        
        # Calculate basic statistics
        max_elev = float(dem_subset.max().values)
        mean_elev = float(dem_subset.mean().values)
        
        # Calculate elevation difference
        elev_diff = max_elev - point_elev
        
        # Simple heuristic for shading potential
        # If there are significantly higher elevations nearby, there's potential for shading
        shading_potential = "Low"
        if elev_diff > 100:
            shading_potential = "Moderate"
        if elev_diff > 300:
            shading_potential = "High"
            
        return {
            "point_elevation": point_elev,
            "max_elevation_nearby": max_elev,
            "mean_elevation_nearby": mean_elev,
            "elevation_difference": elev_diff,
            "shading_potential": shading_potential
        }
    except Exception as e:
        print(f"Error analyzing terrain shading: {e}")
        return {"error": str(e)}


def ensure_crs_alignment(radiation_data, dem_data):
    """
    Ensure that radiation data and DEM have compatible coordinate reference systems
    
    Parameters:
    -----------
    radiation_data : earthkit data object or xarray.Dataset
        The loaded radiation data
    dem_data : xarray.DataArray
        The loaded DEM data
        
    Returns:
    --------
    tuple : (radiation_data, dem_data) with aligned CRS
    """
    print("Checking CRS alignment between datasets...")
    
    # Convert radiation data to xarray if it's not already
    if hasattr(radiation_data, 'to_xarray'):
        rad_xr = radiation_data.to_xarray()
    else:
        rad_xr = radiation_data
    
    # Check if DEM has CRS information
    if hasattr(dem_data, 'rio') and hasattr(dem_data.rio, 'crs'):
        dem_crs = dem_data.rio.crs
        print(f"DEM CRS: {dem_crs}")
        
        # Check if radiation data has CRS information
        rad_crs = None
        if hasattr(rad_xr, 'rio') and hasattr(rad_xr.rio, 'crs'):
            rad_crs = rad_xr.rio.crs
            print(f"Radiation data CRS: {rad_crs}")
        
        # If radiation data doesn't have CRS but has lat/lon, assume it's EPSG:4326
        if rad_crs is None:
            print("Radiation data CRS not found, assuming EPSG:4326 (WGS84)")
            
            # If DEM is not in EPSG:4326, reproject it
            if dem_crs and dem_crs != 'EPSG:4326':
                print(f"Reprojecting DEM from {dem_crs} to EPSG:4326")
                dem_data = dem_data.rio.reproject("EPSG:4326")
        
        # If both have CRS but they're different, reproject DEM to match radiation data
        elif rad_crs and dem_crs and rad_crs != dem_crs:
            print(f"Reprojecting DEM from {dem_crs} to {rad_crs}")
            dem_data = dem_data.rio.reproject(rad_crs)
    
    return radiation_data, dem_data


def main():
    """
    Main function to run the solar radiation and terrain analysis
    """
    # File paths
    radiation_file = "radiation_data.nc"  # or "cams_radiation_data.grib"
    dem_file = "EU_DEM.tif"
    
    # Coordinates for analysis (example: somewhere in Europe)
    latitude = 45.0
    longitude = 10.0
    
    # Variable name for radiation (adjust based on your data)
    # Common names: 'ghi' (Global Horizontal Irradiance), 'dni' (Direct Normal Irradiance)
    radiation_var = 'ghi'
    
    # Load data
    radiation_data = load_radiation_data(radiation_file)
    dem_data = load_dem_data(dem_file)
    
    # Ensure CRS alignment
    radiation_data, dem_data = ensure_crs_alignment(radiation_data, dem_data)
    
    # Extract data at the specified coordinates
    elevation = extract_elevation(dem_data, latitude, longitude)
    radiation_df = extract_radiation_time_series(radiation_data, latitude, longitude)
    
    # Print basic information
    print(f"\nAnalysis at coordinates: {latitude}, {longitude}")
    print(f"Elevation: {elevation:.1f} meters")
    print(f"Available radiation data variables: {list(radiation_df.columns)}")
    
    # Calculate multi-day averages (optional)
    radiation_df = calculate_multi_day_averages(radiation_df, radiation_var, window_days=7)
    
    # Analyze terrain shading potential (optional)
    shading_info = analyze_terrain_shading_potential(dem_data, latitude, longitude)
    print("\nTerrain shading analysis:")
    for key, value in shading_info.items():
        print(f"  {key}: {value}")
    
    # Plot the results
    plot_radiation_time_series(
        radiation_df, 
        radiation_var, 
        elevation, 
        latitude, 
        longitude,
        output_path="solar_radiation_analysis.png"
    )
    
    # If we have multi-day averages, plot those too
    avg_var = f"{radiation_var}_7day_avg"
    if avg_var in radiation_df.columns:
        plot_radiation_time_series(
            radiation_df, 
            avg_var, 
            elevation, 
            latitude, 
            longitude,
            output_path="solar_radiation_7day_avg.png"
        )
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
