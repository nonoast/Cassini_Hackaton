import pygrib
import matplotlib.pyplot as plt

# Load the GRIB file
file_path = 'b7ff4722259a3d467121c416675ce9e2.grib'  # Replace with your GRIB file path
grbs = pygrib.open(file_path)

# Extract the first message (you can adjust the index as needed)
grb = grbs[1]

# Extract the data
data = grb.data()

# Plot the data
plt.figure(figsize=(14, 6))
plt.imshow(data[0], cmap='viridis')
plt.colorbar(label='Value')
plt.title(grb.name)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Close the GRIB file
grbs.close()
