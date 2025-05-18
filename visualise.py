import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from datetime import datetime, timedelta
import cmocean  # for better colormaps
import pandas as pd  # for datetime handling
import os
from matplotlib.colors import LinearSegmentedColormap

def load_dataset(file_path):
    """Load and process NetCDF data."""
    ds = xr.open_dataset(file_path)
    return ds

def create_custom_colormap(base_cmap, alpha=0.7, n_bins=256):
    """Create a custom colormap with adjusted transparency and smoothness."""
    # Get the base colormap
    cmap = plt.cm.get_cmap(base_cmap, n_bins)
    
    # Create new colormap with adjusted alpha
    colors = cmap(np.linspace(0, 1, n_bins))
    colors[:, -1] = alpha  # Adjust alpha channel
    
    # Create new colormap
    return LinearSegmentedColormap.from_list(f'{base_cmap}_custom', colors)

def create_spatial_plot(data, ax, title, cmap='viridis', units='', vmin=None, vmax=None, alpha=0.7):
    """Create a spatial plot for any variable."""
    # If data has more than 2 dimensions, take the first time step
    if len(data.dims) > 2:
        time_dim = next((dim for dim in data.dims if 'time' in dim.lower()), None)
        if time_dim:
            data = data.isel({time_dim: 0})  # Take first time step
            print(f"Using first time step from {time_dim} dimension")
    
    # Get longitude and latitude values
    lon_name = next((dim for dim in data.dims if 'lon' in dim.lower()), 'longitude')
    lat_name = next((dim for dim in data.dims if 'lat' in dim.lower()), 'latitude')
    
    # Print data information for debugging
    print(f"\nData shape: {data.shape}")
    print(f"Dimensions: {data.dims}")
    print(f"Coordinate names: lon={lon_name}, lat={lat_name}")
    
    lon = data[lon_name].values
    lat = data[lat_name].values
    
    # Create 2D meshgrid for coordinates
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    
    # Calculate vmin and vmax if not provided
    if vmin is None:
        vmin = np.percentile(data.values, 5)  # 5th percentile for smoother range
    if vmax is None:
        vmax = np.percentile(data.values, 95)  # 95th percentile for smoother range
    
    # Ensure data is 2D
    plot_data = data.values
    if len(plot_data.shape) > 2:
        plot_data = plot_data.squeeze()  # Remove single-dimensional entries
    
    # Create custom colormap
    custom_cmap = create_custom_colormap(cmap, alpha=alpha)
    
    # Plot the data
    im = ax.pcolormesh(lon_mesh, lat_mesh, plot_data,
                      transform=ccrs.PlateCarree(),
                      cmap=custom_cmap,
                      vmin=vmin,
                      vmax=vmax,
                      shading='auto')
    
    # Add coastlines and gridlines with reduced opacity
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, alpha=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    # Add colorbar with adjusted appearance
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05,
                       label=f'{title} ({units})')
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.5)
    
    # Set title with reduced font weight
    ax.set_title(title, fontsize=12, pad=10, fontweight='normal')
    return im

def create_temperature_profile(ds, variable, target_lat, target_lon, ax):
    """Create a profile plot showing temperature along latitude and longitude."""
    data = ds[variable]
    
    # If data has more than 2 dimensions, take the first time step
    if len(data.dims) > 2:
        time_dim = next((dim for dim in data.dims if 'time' in dim.lower()), None)
        if time_dim:
            data = data.isel({time_dim: 0})  # Take first time step
    
    # Convert to Celsius if the data is in Kelvin
    if 'units' in data.attrs and 'K' in data.attrs['units']:
        data = data - 273.15
        units = '°C'
    else:
        units = data.attrs.get('units', '°C')
    
    # Handle different possible dimension names
    lon_name = next((dim for dim in data.dims if 'lon' in dim.lower()), 'longitude')
    lat_name = next((dim for dim in data.dims if 'lat' in dim.lower()), 'latitude')
    
    # Find nearest grid point
    lat_idx = np.abs(data[lat_name].values - target_lat).argmin()
    lon_idx = np.abs(data[lon_name].values - target_lon).argmin()
    
    # Plot latitude profile
    lat_profile = data.isel({lon_name: lon_idx})
    ax.plot(lat_profile[lat_name], lat_profile, 
            label=f'Latitude profile at {target_lon}°E',
            color='red', linewidth=1.5)
    
    # Plot longitude profile
    lon_profile = data.isel({lat_name: lat_idx})
    ax.plot(lon_profile[lon_name], lon_profile, 
            label=f'Longitude profile at {target_lat}°N',
            color='blue', linewidth=1.5)
    
    ax.set_xlabel('Latitude/Longitude')
    ax.set_ylabel(f'Temperature ({units})')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Add some statistics
    mean_temp = data.values.mean()
    std_temp = data.values.std()
    ax.text(0.02, 0.98, 
            f'Mean: {mean_temp:.1f}{units}\nStd: {std_temp:.1f}{units}',
            transform=ax.transAxes, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=8)

def main():
    # Define dataset paths
    dataset_dir = 'Datasets'
    datasets = {
        'Temperature': {
            'file': os.path.join(dataset_dir, 'temperature.nc'),
            'variable': 't2m',
            'title': '2-meter Temperature',
            'cmap': 'RdBu_r',
            'units': '°C',
            'alpha': 0.6,  # More transparent
            'process_func': lambda x: x - 273.15 if 'units' in x.attrs and 'K' in x.attrs['units'] else x
        },
        'Cloud Cover': {
            'file': os.path.join(dataset_dir, 'cloud_cover.nc'),
            'variable': 'tcc',
            'title': 'Total Cloud Cover',
            'cmap': 'Blues',
            'units': '%',
            'alpha': 0.5,  # More transparent
            'process_func': lambda x: x * 100 if 'units' not in x.attrs else x
        },
        'Radiation': {
            'file': os.path.join(dataset_dir, 'radiation.nc'),
            'variable': 'cdir',
            'title': 'Surface Solar Radiation',
            'cmap': 'YlOrRd',
            'units': 'W/m²',
            'alpha': 0.6,  # More transparent
            'process_func': lambda x: x
        }
    }
    
    # Create a figure with three subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Set figure background to white
    fig.patch.set_facecolor('white')
    
    # Load and plot each dataset
    for idx, (name, info) in enumerate(datasets.items(), 1):
        print(f"\nProcessing {name}...")
        
        # Load dataset
        ds = load_dataset(info['file'])
        
        # Print dataset information
        print(f"\n{name} Dataset Information:")
        print("Variables:", list(ds.data_vars))
        print("Dimensions:", ds.dims)
        
        # Get the variable data
        data = ds[info['variable']]
        
        # Process the data (e.g., convert units)
        data = info['process_func'](data)
        
        # Create subplot
        ax = fig.add_subplot(2, 2, idx, projection=ccrs.PlateCarree())
        
        # Create the plot with custom alpha
        create_spatial_plot(data, ax, info['title'], 
                          cmap=info['cmap'], 
                          units=info['units'],
                          alpha=info['alpha'])
    
    # Add a title for the entire figure with reduced weight
    fig.suptitle('Global Meteorological Data Visualization', 
                 fontsize=16, y=0.95, fontweight='normal')
    
    # Adjust layout and save with high DPI for smoothness
    plt.tight_layout()
    plt.savefig('meteorological_data.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    # Print summary of created visualization
    print("\nVisualization created: meteorological_data.png")
    print("The figure contains:")
    for name, info in datasets.items():
        print(f"- {info['title']} ({info['units']})")

if __name__ == "__main__":
    main()
