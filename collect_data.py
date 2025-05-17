import cdsapi

client = cdsapi.Client()

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": ["clear_sky_direct_solar_radiation_at_surface"],
    "year": ["2025"],
    "month": [
        "01", "02", "03",
        "04", "05"
    ],
    "day": ["01"],
    "time": ["12:00"],
    "data_format": "grib",
    "download_format": "unarchived"
}

client.retrieve(dataset, request).download()

print("Téléchargement terminé.")


