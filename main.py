from raster_clipper import RasterClipper
from common import lines_to_polygon
from clusterable import NDVIData


if __name__ == '__main__':
    ndvi = NDVIData("/home/dverholancev/study/degree/src/20230608_F14_Micasense_NDVI.tif")
    ndvi.load()
    # ndvi.clean()

    clipper = RasterClipper(
        ndvi,
        "/home/dverholancev/study/degree/src/boundary_field_14.shp",
        'latin1',
        lines_to_polygon)
    clipped_ndvi = clipper.clip()
    clipped_ndvi.save('clipped.tif')