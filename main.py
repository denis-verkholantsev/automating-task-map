from raster_clipper import RasterClipper
from common import lines_to_polygon


if __name__ == '__main__':
    clipper = RasterClipper(
        source_path="/home/dverholancev/study/degree/src/20230608_F14_Micasense_NDVI.tif",
        boundary_path="/home/dverholancev/study/degree/src/boundary_field_14.shp",
        output_path="clippedd.tif",
        encoding='latin1',
        preprocess_geometry=lines_to_polygon
    )
    clipper.clip()
