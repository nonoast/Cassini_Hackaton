import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "clear_sky_direct_solar_radiation_at_surface",
        "downward_uv_radiation_at_the_surface",
        "uv_visible_albedo_for_diffuse_radiation",
        "uv_visible_albedo_for_direct_radiation",
        "total_cloud_cover"
    ],
    "year": ["2025"],
    "month": ["01"],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20"
    ],
    "time": [
        "07:00", "08:00", "09:00",
        "10:00", "11:00", "12:00",
        "13:00", "14:00", "15:00",
        "16:00", "17:00", "18:00",
        "19:00", "20:00", "21:00"
    ],
    "data_format": "grib",
    "download_format": "unarchived"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()


print("Téléchargement terminé.")


