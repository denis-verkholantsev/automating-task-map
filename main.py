from raster_clipper import RasterClipper
from common import lines_to_polygon, analyze_ndvi, get_ndvi_stats_by_ranges
from clusterable import NDVIData
from cluster import KMeansRasterClustering


if __name__ == '__main__':
    ndvi = NDVIData.load("/home/dverholancev/study/degree/src/20230608_F14_Micasense_NDVI.tif")
    # ndvi.load()
    # ndvi.clean()
    # analyze_ndvi(ndvi.data)

    clipper = RasterClipper(
        ndvi,
        "/home/dverholancev/study/degree/src/boundary_field_14.shp",
        'latin1',
        lines_to_polygon)

    clipped_ndvi = clipper.clip()
    clipped_ndvi.clean()
    # analyze_ndvi(clipped_ndvi.data)
    # get_ndvi_stats_by_ranges(clipped_ndvi.data)
    clipped_ndvi.save('clippedd.tif')
    result = KMeansRasterClustering.fit(clipped_ndvi, 3, 50)
    KMeansRasterClustering.export_shapefile(result, 'cluster_with_50.shp', clipped_ndvi.transform, clipped_ndvi.crs)
    KMeansRasterClustering.save_clustered_raster(result, 'clustered.tif', clipped_ndvi.profile)
