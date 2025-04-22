import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from typing import Callable
import numpy as np


class RasterClipper:
    def __init__(
            self, 
            source_path: str, 
            boundary_path: str, 
            output_path: str, 
            encoding: str = 'utf-8', 
            preprocess_geometry: Callable[[str, str], None] | None = None
    ):
        self.source_path = source_path
        self.boundary_path = boundary_path
        self.output_path = output_path
        self.encoding = encoding
        self.geometry = self._load_geometry(preprocess_geometry)

    def _load_geometry(self, preprocess_geometry):
        if preprocess_geometry:
            self.boundary_path = preprocess_geometry(self.boundary_path, self.encoding)

        with rasterio.open(self.source_path) as src:
            raster_crs = src.crs

        gdf = gpd.read_file(self.boundary_path, encoding=self.encoding)

        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        return [mapping(geom) for geom in gdf.geometry if not geom.is_empty]
        #return [geom.__geo_interface__ for geom in gdf.geometry.values]

    def clip(self):
        with rasterio.open(self.source_path) as src:
            out_image, out_transform = mask(src, self.geometry, crop=True)
            out_meta = src.meta.copy()
            raster_crs = src.crs
        
        out_image = np.where(out_image == -32767.0, 0, out_image)
        out_meta["nodata"] = 0
        
        print(out_image.shape) 
        print(out_image.min(), out_image.max())

        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": raster_crs
        })

        with rasterio.open(self.output_path, "w", **out_meta) as dest:
            dest.write(out_image)
    