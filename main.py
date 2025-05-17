from raster_clipper import RasterClipper
from common import lines_to_polygon, analyze_ndvi, get_ndvi_stats_by_ranges
from clusterable import NDVIData
from cluster import KMeansRasterClustering


def run_clipped():
    ndvi = NDVIData.load("/home/dverholancev/study/degree/src/cliiped_mini.tif")
    ndvi.clean()
    result = KMeansRasterClustering.fit(ndvi, 5, 1)
    KMeansRasterClustering.export_shapefile(result, 'clustered5_with_1_bfs_most_common.shp', ndvi.transform, ndvi.crs)


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


if __name__ == '__main__':
    run_clipped()