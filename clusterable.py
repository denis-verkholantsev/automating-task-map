import numpy as np
import rasterio


class NDVIData:
    def __init__(self, path: str | None = None):
        self.path = path
        self.data = None
        self.profile = None
        self.crs = None
    
    @staticmethod
    def from_data(image, meta, crs, path=None):
        result = NDVIData(path)
        result.data = image
        result.profile = meta
        result.profile["crs"] = crs
        result.crs = crs
        return result

    def load(self):
        with rasterio.open(self.path) as src:
            self.data = src.read()
            self.profile = src.profile
            self.crs = src.crs
        print(f"📥 NDVI loaded from {self.path}")

    def clean(self):
        self.data = np.where((self.data < -1) | (self.data > 1), np.nan, self.data)
        print("🧹 NDVI cleaned")
    
    def save(self, path: str):
        if self.data is None or self.profile is None:
            raise ValueError("NDVI data or profile not set.")
        with rasterio.open(path, 'w', **self.profile) as dst:
            dst.write(self.data)
        print(f"💾 NDVI saved to {path}")
