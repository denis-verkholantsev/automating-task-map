import geopandas as gpd
import rasterio
import argparse

from rasterio.mask import mask
from shapely.geometry import mapping
from typing import Callable

from clusterable import NDVIData
from common import create_filepath, lines_to_polygon


class RasterClipper:
    def __init__(
            self, 
            ndvi: NDVIData, # source
            boundary_path: str, # required line or polygon if use lines_to_polygon
            encoding: str = 'utf-8', # sometimes required 'latin1'
            preprocess_geometry: Callable[[str, str | None], str] | None = None# transform boundary_path to polygon
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
    
        clipped_ndvi = NDVIData.from_data(out_image, out_meta, self.ndvi.crs, self.ndvi.pixel_width, self.ndvi.pixel_height)
        print("NDVI clipped into new NDVIData")
        return clipped_ndvi


def main():
    parser = argparse.ArgumentParser(description="Clip NDVI raster using vector geometry.")
    parser.add_argument("input", help="Path to NDVI raster (GeoTIFF)")
    parser.add_argument("boundary", help="Path to boundary vector file (shapefile, etc.)")
    parser.add_argument("--output",'-o', required=False, help="Path to save clipped NDVI GeoTIFF")
    parser.add_argument("--encoding", 'e', default="latin1", help="Encoding for reading vector file (default: latin1)")

    args = parser.parse_args()

    ndvi = NDVIData.load(args.input)

    clipper = RasterClipper(
        ndvi=ndvi,
        boundary_path=args.boundary,
        encoding=args.encoding,
        preprocess_geometry=lines_to_polygon
    )

    clipped_ndvi = clipper.clip()
    output_path = args.output or create_filepath(args.input, 'tif', 'clipped')
    clipped_ndvi.save(output_path)
    print(f"Saved clipped NDVI raster to {output_path}")


if __name__ == "__main__":
    main()
