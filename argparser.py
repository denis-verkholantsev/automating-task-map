import argparse

def call(args):

    ndvi = NDVIData.load(args.input)

    # Очистка NDVI
    if args.clean:
        ndvi.clean()

    # Сжатие
    if args.compress:
        ndvi.compress(tuple(args.compress), pixel=args.pixel)

    # Сохранение обработанного GeoTIFF
    ndvi.save(args.output)

    # Экспорт в shapefile
    if args.export:
        if not args.shapefile:
            raise ValueError("Не указан путь к shapefile. Используйте --shapefile")
        BaseRasterClustering.export_shapefile(ndvi.data, args.shapefile, ndvi.transform, ndvi.crs)



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
        "--cluster_method",
        type=str,
        choices=[
            "most_common_label",
            "bfs_most_common",
            "bfs_nearest",
            "label_most_common",
            "label_nearest"
        ],
        required=True,
        help="Метод агрегации кластеров"
    )

    parser.add_argument(
        "--min_area",
        type=float,
        default=100.0,
        help="Минимальная площадь для фильтрации малых кластеров (в м²)"
    )

    parser.add_argument(
        "--compress_block",
        type=float,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        help="Сжать NDVI до блока (в метрах)"
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Очистить NDVI: оставить только значения от 0 до 1"
    )

    return parser.parse_args()
