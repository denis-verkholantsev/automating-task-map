from raster_clipper import RasterClipper
from common import lines_to_polygon, decompress_array, create_filepath, create_fertilizer_shapefile
from clusterable import NDVIData
from cluster import KMeansRasterClustering, HDBSCANRasterClustering, BaseRasterClustering
from config import Config
from grid import block_average_with_edges, export_shapefile

MAX_SIZE = 50000000

# PHRASE_TO_CLEAN_METHOD = {
#     'most_common_label': '_process_block_with_most_common_label',
#     'bfs_most_common': '_process_block_bfs_with_most_common_cluster',
#     'bfs_nearest': '_process_block_bfs_with_nearest_cluster',
#     'label_most_common': '_process_block_label_with_most_common_label',
#     'label_nearest': '_process_block_label_with_nearest_label',
# }

def run_clipped(config):
    ndvi = NDVIData.load("/home/dverholancev/study/degree/app/ndvi_output.tif")
    clipper = RasterClipper(
        ndvi,
        "/home/dverholancev/study/degree/src/boundary_field_14.shp",
        'latin1',
        lines_to_polygon)

    ndvi = clipper.clip()
    ndvi.clean()
    ndvi.compress((1, 1))
    result, stats = KMeansRasterClustering.fit(ndvi, 5, 8, clean_method='label_nearest', block_size=(15, 15), workers=config.WORKERS)
    block = ndvi.compression_block
    ndvi.decompress()
    result = decompress_array(result, block)
    KMeansRasterClustering.export_shapefile(result, 'clustered_compressed_8.shp', ndvi.transform, ndvi.crs)

def run_compress():
    ndvi = NDVIData.load("/home/dverholancev/study/degree/app/ndvi_output.tif")
    clipper = RasterClipper(
        ndvi,
        "/home/dverholancev/study/degree/src/boundary_field_14.shp",
        'latin1',
        lines_to_polygon)

    ndvi = clipper.clip()
    ndvi.clean()
    ndvi.save('clipped_last.tif')
    result, stats = KMeansRasterClustering.fit(ndvi, 5, 5, clean_method='label_nearest', block_size=(15, 15), workers=config.WORKERS)
    KMeansRasterClustering.export_shapefile(result, 'clustered_new_3_15x15_label_nearest.shp', ndvi.transform, ndvi.crs, stats=stats)
    create_fertilizer_shapefile('clustered_new_3_15x15_label_nearest.shp', 'out_new.shp',2, 60)

def run(config):
    ndvi = NDVIData.load("/home/dverholancev/study/degree/src/20230608_F14_Micasense_NDVI.tif")
    clipper = RasterClipper(
        ndvi,
        "/home/dverholancev/study/degree/src/boundary_field_14.shp",
        'latin1',
        lines_to_polygon)

    clipped_ndvi = clipper.clip()
    clipped_ndvi.clean()
    clipped_ndvi.save('clippedd.tif')
    # result = KMeansRasterClustering.fit(clipped_ndvi, 5, 10)
    result = KMeansRasterClustering.fit(clipped_ndvi, 4, 3, clean_method='label_nearest', block_size=(10, 10), workers=config.WORKERS, use_mini_batch=True)
    KMeansRasterClustering.export_shapefile(result, 'clustered_3_10x10_full.shp', clipped_ndvi.transform, clipped_ndvi.crs)


def run_hdbscan():
    ndvi = NDVIData.load("/home/dverholancev/study/degree/src/cliiped_mini.tif")
    ndvi.clean()
    ndvi.compress((0.2, 0.2))
    result = HDBSCANRasterClustering.fit(ndvi, min_cluster_area=1000, min_samples=3, workers=-1)
    block = ndvi.compression_block
    ndvi.decompress()
    result = decompress_array(result, block)
    KMeansRasterClustering.export_shapefile(result, 'hdbscan_clustered_3.shp', ndvi.transform, ndvi.crs)


def run_simple_grid():
    ndvi = NDVIData.load("/home/dverholancev/study/degree/src/20230608_F14_Micasense_NDVI.tif")
    clipper = RasterClipper(
        ndvi,
        "/home/dverholancev/study/degree/src/boundary_field_14.shp",
        'latin1',
        lines_to_polygon)

    clipped_ndvi = clipper.clip()
    clipped_ndvi.clean()
    clipped_ndvi.save('clippedd.tif')
    result = block_average_with_edges(clipped_ndvi, block_size=(10, 10))
    export_shapefile(result, 'grid_10x10.shp', clipped_ndvi.transform, clipped_ndvi.crs)


config = Config()
# run_clipped(config)
# run_hdbscan()
run_clipped(config)

import argparse

def call(args):
    ndvi = NDVIData.load(args.input)

    if args.border:
        clipper = RasterClipper(
            ndvi,
            args.border,
            'latin1',
            lines_to_polygon)

        ndvi = clipper.clip()
        if args.save:
            ndvi.save(create_filepath(args.input))

    to_decompress = False
    if args.compress_block:
        ndvi.compress(args.compress_block, pixel=args.in_pixels)
        to_decompress = True

    result, stats = None, None
    if args.cluster_method == 'k-means':
        use_mini_batch = True if ndvi.size() > 50000000 else False
        n_clusters = args.clusters_number
        min_cluster_area = args.min_area
        postprocessing_method = args.postprocessing_method
        work_block = args.work_block
        result, stats = KMeansRasterClustering.fit(
            ndvi,
            n_clusters=n_clusters,
            min_cluster_area=min_cluster_area,
            clean_method=postprocessing_method,
            block_size=work_block,
            use_mini_batch=use_mini_batch,
            workers=config.WORKERS
        )
    elif args.cluster_method == 'hdbscan':
        min_cluster_area = args.min_area
        min_samples = args.min_samples
        result = HDBSCANRasterClustering.fit(
            ndvi,
            min_cluster_area=min_cluster_area,
            min_samples=min_samples,
            workers=config.WORKERS
        )

    if to_decompress:
        block = ndvi.compression_block
        ndvi.decompress()
        result = decompress_array(result, block)

    if args.export:
        BaseRasterClustering.export_shapefile(
            result,
            args.shapefile or create_filepath(args.input, args.cluster_method),
            ndvi.transform,
            ndvi.crs,
            stats=stats
        )

def parse_args():
    parser = argparse.ArgumentParser(description="NDVI clustering tool")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Путь к NDVI GeoTIFF-файлу"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Путь к выходному Shapefile (например, output.shp)"
    )

    parser.add_argument(
        "--border",
        type=str,
        help="Путь к файлу границы .shp (lines/polygon) (например, border.shp)"
    )

    parser.add_argument(
        "--save_clipped",
        type=bool,
        help="Сохранить обрезанный tif"
    )

    parser.add_argument(
        "--cluster_method",
        type=str,
        choices=[
            "k-means",
            "hdbscan",
        ],
        required=True,
        help="Метод кластеризации"
    )

    parser.add_argument(
        "--clusters_number",
        type=int,
        help="Количество кластеров"
    )

    parser.add_argument(
        "--min_cluster_size",
        type=float,
        help="Минимальная площадь кластера в блоке в м2"
    )

    parser.add_argument(
        "--work_block",
        action="store_true",
        help="Обработка блоков"
    )

    parser.add_argument(
        "--postprocessing_method",
        type=str,
        choices=[
            "most_common_label",
            "bfs_most_common",
            "bfs_nearest",
            "label_most_common",
            "label_nearest"
        ],
        required=True,
        help="Метод агрегации областей кластеров"
    )

    parser.add_argument(
        "--min_area",
        type=float,
        default=100.0,
        help="Минимальная площадь для фильтрации малых кластеров (в м²)"
    )

    parser.add_argument(
        "--min_samples",
        type=int,
        help="Минимальное количество соседей"
    )

    parser.add_argument(
        "--compress_block",
        type=float,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        help="Сжать NDVI до блока (в метрах)"
    )

    parser.add_argument(
        "--in_pixels",
        type=bool,
        help="Путь к файлу границы .shp (lines/polygon) (например, border.shp)"
    )


    return parser.parse_args()
