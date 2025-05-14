import numpy as np
import rasterio
from common import convert_pixel_size_to_meters


class NDVIData:
    def __init__(self, path: str | None = None):
        self._path = path
        self._data = None
        self._profile = None
        self._crs = None
        self._pixel_area = None
        self._pixel_width = None
        self._pixel_height = None
    
    @staticmethod
    def from_data(image, meta, crs, pixel_width, pixel_height, path=None) -> 'NDVIData':
        result = NDVIData(path)
        result._data = image
        result._profile = meta
        result._profile["crs"] = crs
        result._crs = crs
        result._pixel_width = pixel_width
        result._pixel_height = pixel_height

        return result
    
    @staticmethod
    def load(path: str) -> 'NDVIData':
        result = NDVIData(path)
        with rasterio.open(path) as src:
            result._data = src.read()
            result._profile = src.profile
            result._crs = src.crs
            result._pixel_width, result._pixel_height = src.res
            result._pixel_area = abs(result._pixel_width * result._pixel_height)

        if result._crs.to_epsg() == 4326:
            center_lon = (src.bounds.left + src.bounds.right) / 2
            center_lat = (src.bounds.top + src.bounds.bottom) / 2

            pixel_width_m, pixel_height_m = convert_pixel_size_to_meters(
                result._pixel_width, result._pixel_height,
                center_lon, center_lat
            )

            result._pixel_width = pixel_width_m
            result._pixel_height = pixel_height_m
            result._pixel_area = pixel_width_m * pixel_height_m
        else:
            result._pixel_area = abs(result._pixel_width * result._pixel_height)

        print(f"Converted pixel size to meters: {result._pixel_width} x {result._pixel_height} meters.")
        print(f"NDVI loaded from {result._path}")

        return result
    
    @property
    def data(self):
        return self._data
    
    @property
    def crs(self):
        return self._crs
    
    @property
    def path(self):
        return self._path
    
    @property
    def profile(self):
        return self._profile
    
    @property
    def transform(self):
        return self._profile.get('transform')
    
    @property
    def pixel_width(self):
        return self._pixel_width
    
    @property
    def pixel_height(self):
        return self._pixel_height

    def clean(self): 
        self._data = np.where((self._data <= 0) | (self._data > 1), np.nan, self._data)
        print("NDVI cleaned")
    
    def save(self, path: str):
        if self._data is None or self._profile is None:
            raise ValueError("NDVI data or profile not set.")
        
        with rasterio.open(path, 'w', **self._profile) as dst:
            dst.write(self._data)
        print(f"NDVI saved to {path}")
