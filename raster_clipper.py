import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from typing import Callable
from clusterable import NDVIData


class RasterClipper:
    def __init__(
            self, 
            ndvi: NDVIData, 
            boundary_path: str, 
            encoding: str = 'utf-8', 
            preprocess_geometry: Callable[[str, str | None], None] | None = None
    ):
        self.ndvi = ndvi
        self.boundary_path = boundary_path
        self.encoding = encoding
        self.geometry = self._load_geometry(preprocess_geometry)

    def _load_geometry(self, preprocess_geometry):
        if preprocess_geometry:
            self.boundary_path = preprocess_geometry(self.boundary_path, self.encoding)

        gdf = gpd.read_file(self.boundary_path, encoding=self.encoding)

        if gdf.crs != self.ndvi.crs:
            gdf = gdf.to_crs(self.ndvi.crs)

        return [mapping(geom) for geom in gdf.geometry if not geom.is_empty]

    def clip(self) -> NDVIData:
        with rasterio.io.MemoryFile() as memfile:
           with memfile.open(**self.ndvi.profile) as tmp:
               tmp.write(self.ndvi.data)
               out_image, out_transform = mask(tmp, self.geometry, crop=True)
               out_meta = tmp.meta.copy()
        
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        })
    
        clipped_ndvi = NDVIData.from_data(out_image, out_meta, crs=self.ndvi.crs)
        print("✂️ NDVI clipped into new NDVIData")
        return clipped_ndvi
    