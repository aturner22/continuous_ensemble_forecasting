import xarray as xr
import os

zarr_path = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
output_dir = "era5_data"
os.makedirs(output_dir, exist_ok=True)

# Required variables mapping: output_name → (source_name, optional_level)
variable_map = {
    "10m_u_component_of_wind": ("10m_u_component_of_wind", None),
    "10m_v_component_of_wind": ("10m_v_component_of_wind", None),
    "2m_temperature": ("2m_temperature", None),
    "geopotential_500": ("geopotential", 500),
    "temperature_850": ("temperature", 850),
}

# Load Zarr with lazy access
ds = xr.open_zarr(zarr_path, consolidated=True)

# DEMO CONFIGURATION: Download more time steps for viable testing
# Original: 100 time steps → New: 500 time steps
# This gives us: 400 train, 50 val, 50 test samples
DEMO_TIME_STEPS = 2000

print(f"Downloading {DEMO_TIME_STEPS} time steps for demo...")
print("This will create:")
print(f"  - Training samples: {int(0.8 * DEMO_TIME_STEPS)}")
print(f"  - Validation samples: {int(0.1 * DEMO_TIME_STEPS)}")
print(f"  - Test samples: {DEMO_TIME_STEPS - int(0.8 * DEMO_TIME_STEPS) - int(0.1 * DEMO_TIME_STEPS)}")

# Extract constants: orography from geopotential at surface level; land_sea_mask
const_dir = os.path.join(output_dir, "constants")
os.makedirs(const_dir, exist_ok=True)

# Orography is surface geopotential divided by g
if "geopotential" not in ds:
    raise RuntimeError("Missing 'geopotential' field in ERA5 Zarr.")

if "land_sea_mask" not in ds:
    raise RuntimeError("Missing 'land_sea_mask' field in ERA5 Zarr.")

lat = ds.latitude.values
lon = ds.longitude.values

static = xr.Dataset()
static["lat"] = (("latitude",), lat)
static["lon"] = (("longitude",), lon)

# Take first available time and level (assumes time-invariant)
static["orography"] = ds["geopotential"].isel(time=0, level=0) / 9.80665
static["lsm"] = ds["land_sea_mask"]

static.to_netcdf(os.path.join(const_dir, "constants.nc"))
print("✓ Constants saved")

# Save dynamic fields with more time steps
for output_name, (source_name, level) in variable_map.items():
    var_dir = os.path.join(output_dir, output_name)
    os.makedirs(var_dir, exist_ok=True)

    if source_name not in ds:
        raise KeyError(f"Variable '{source_name}' not found in dataset.")

    print(f"Processing {output_name}...")
    
    if level is None:
        # Changed from slice(0, 100) to slice(0, DEMO_TIME_STEPS)
        data = ds[source_name].isel(time=slice(0, DEMO_TIME_STEPS))
        output_ds = xr.Dataset({output_name: data})
    else:
        if "level" not in ds[source_name].dims:
            raise ValueError(f"Variable '{source_name}' does not have a 'level' dimension.")
        if level not in ds["level"]:
            raise ValueError(f"Level {level} not found in variable '{source_name}'.")
        # Changed from slice(0, 100) to slice(0, DEMO_TIME_STEPS)
        data = ds[source_name].sel(level=level).isel(time=slice(0, DEMO_TIME_STEPS))
        output_ds = xr.Dataset({output_name: data})

    path = os.path.join(var_dir, f"{output_name}.nc")
    output_ds.to_netcdf(path)
    print(f"✓ {output_name} saved")

print(f"\n✅ Demo dataset download complete!")
print(f"Downloaded {DEMO_TIME_STEPS} time steps for each variable")
print(f"Expected final dataset split:")
print(f"  - Training: {int(0.8 * DEMO_TIME_STEPS)} samples")
print(f"  - Validation: {int(0.1 * DEMO_TIME_STEPS)} samples") 
print(f"  - Test: {DEMO_TIME_STEPS - int(0.8 * DEMO_TIME_STEPS) - int(0.1 * DEMO_TIME_STEPS)} samples")
print(f"\nThis should provide enough test samples for your forecasting configuration.")