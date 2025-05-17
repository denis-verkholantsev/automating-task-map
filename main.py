from raster_clipper import RasterClipper
from common import lines_to_polygon
from clusterable import NDVIData
from cluster import KMeansRasterClustering
from config import Config


# PHRASE_TO_CLEAN_METHOD = {
#     'most_common_label': '_process_block_with_most_common_label',
#     'bfs_most_common': '_process_block_bfs_with_most_common_cluster',
#     'bfs_nearest': '_process_block_bfs_with_nearest_cluster',
#     'label_most_common': '_process_block_label_with_most_common_label',
#     'label_nearest': '_process_block_label_with_nearest_label',
# }

def run_clipped(config):
    ndvi = NDVIData.load("/home/dverholancev/study/degree/src/cliiped_mini.tif")
    ndvi.clean()
    result = KMeansRasterClustering.fit(ndvi, 5, 1, clean_method='bfs_most_common', block_size=(4, 4), workers=config.WORKERS)
    KMeansRasterClustering.export_shapefile(result, 'clustered_1_bfs_most_common_label.shp', ndvi.transform, ndvi.crs)


def run():
    ndvi = NDVIData.load("/home/dverholancev/study/degree/src/20230608_F14_Micasense_NDVI.tif")
    clipper = RasterClipper(
        ndvi,
        "/home/dverholancev/study/degree/src/boundary_field_14.shp",
        'latin1',
        lines_to_polygon)

    clipped_ndvi = clipper.clip()
    clipped_ndvi.clean()
    clipped_ndvi.save('clippedd.tif')
    result = KMeansRasterClustering.fit(clipped_ndvi, 5, 10)
    KMeansRasterClustering.export_shapefile(result, 'cluster_with_400.shp', clipped_ndvi.transform, clipped_ndvi.crs)


config = Config()
run_clipped(config)