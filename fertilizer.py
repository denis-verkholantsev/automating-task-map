import geopandas as gpd
import pandas as pd
import argparse
import numpy as np

from common import create_filepath


def required_units_sum(gdf):
    gdf = gdf.to_crs("EPSG:32637")
    gdf['area_m2'] = gdf.geometry.area
    gdf['total_fert'] = gdf['fertiliz'] * gdf['area_m2']

    total_sum = gdf['total_fert'].sum()
    print(f"Total fertilizer required: {total_sum:.2f} units")


def create_fertilizer_shapefile_manual(
        input_shp: str,
        output_shp: str,
        fertilizer_by_cluster: dict[int, float],
        input_gdf: gpd.GeoDataFrame | None = None
) -> None:
    print(f"Loading shapefile: {input_shp}")
    gdf = input_gdf if input_gdf is not None else gpd.read_file(input_shp)

    if 'cluster_id' not in gdf.columns:
        raise ValueError("Input shapefile must have 'cluster_id' column")

    def get_fertilizer(row):
        cluster_id = int(row['cluster_id'])
        fert = fertilizer_by_cluster.get(cluster_id, 0)  # 0 если кластер не указан
        return fert

    gdf['fertiliz'] = gdf.apply(get_fertilizer, axis=1)
    required_units_sum(gdf)

    gdf.to_file(output_shp)
    print(f"Saved fertilizer shapefile to {output_shp}")


def create_fertilizer_shapefile_by_cluster_id(
        input_shp: str,
        output_shp: str,
        target_cluster_id: int,
        target_fertilizer: float,
        input_gdf: gpd.GeoDataFrame | None = None
) -> None:
    gdf = input_gdf if input_gdf is not None else gpd.read_file(input_shp)

    if 'cluster_id' not in gdf.columns:
        raise ValueError("Input shapefile must have 'cluster_id' and 'mean' columns")

    #  cluster_id -> mean
    stats = {
        int(row['cluster_id']): float(row['mean'])
        for _, row in gdf.iterrows()
        if not pd.isna(row['cluster_id']) and not pd.isna(row['mean'])
    }

    base_ndvi = stats.get(target_cluster_id)
    if base_ndvi is None:
        raise ValueError(f"Target cluster_id {target_cluster_id} not found")

    # дозы удобрения
    def calc_fertilizer(cluster_id, ndvi):
        if ndvi is None or np.isnan(ndvi) or ndvi <= 0:
            return 0
        return round(target_fertilizer * (base_ndvi / ndvi), 2)

    #  столбец с дозами удобрений
    def get_fertilizer(row):
        cluster = int(row['cluster_id'])
        ndvi = float(row['mean'])
        return calc_fertilizer(cluster, ndvi)

    gdf['fertiliz'] = gdf.apply(get_fertilizer, axis=1)
    required_units_sum(gdf)

    # в новый shp
    gdf.to_file(output_shp)
    print(f"Saved fertilizer shapefile to {output_shp}")


def create_fertilizer_shapefile_by_ndvi(
        input_shp: str,
        output_shp: str,
        target_ndvi: float,
        target_fertilizer: float,
        input_gdf: gpd.GeoDataFrame | None = None
) -> None:
    gdf = input_gdf if input_gdf is not None else gpd.read_file(input_shp)

    if 'mean' not in gdf.columns:
        raise ValueError("Input shapefile must have 'mean' column")

    def calc_fertilizer(ndvi):
        if ndvi is None or np.isnan(ndvi) or ndvi <= 0:
            return 0
        fert = round(target_fertilizer * (target_ndvi / ndvi), 2)
        return fert

    gdf['fertiliz'] = gdf['mean'].apply(lambda ndvi: calc_fertilizer(float(ndvi)))
    required_units_sum(gdf)

    gdf.to_file(output_shp)
    print(f"Saved fertilizer shapefile to {output_shp}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create fertilizer shapefile")

    parser.add_argument("input", help="Input shapefile path")
    parser.add_argument("--output", "-o", help="Output shapefile path")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manual", nargs="+", metavar="CLUSTER=VALUE",
                       help="Manual fertilizer per cluster, e.g. 1=50.0 2=60.5")
    group.add_argument("--by-cluster-id", type=int,
                       help="Reference cluster_id for automatic calculation (requires --target-fertilizer)")
    group.add_argument("--by-ndvi", type=float,
                       help="Alias of --by-cluster-id")

    parser.add_argument("--target", type=float,
                        help="Fertilizer dose for the reference cluster (required with --by-cluster-id or --by-ndvi)")

    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output or create_filepath(args.input, 'shp', 'fertilizer')
    if args.manual:
        fert_dict = {}
        for item in args.manual:
            try:
                cluster_str, value_str = item.split("=")
                fert_dict[int(cluster_str)] = float(value_str)
            except ValueError:
                raise ValueError(f"Invalid fertilizer format: {item}. Use CLUSTER=VALUE")
        create_fertilizer_shapefile_manual(args.input, output, fert_dict)

    elif args.by_cluster_id is not None:
        if args.target is None:
            raise ValueError("You must specify --target with --by-cluster-id")
        create_fertilizer_shapefile_by_cluster_id(
            args.input, output, args.by_cluster_id, args.target
        )

    elif args.by_ndvi is not None:
        if args.target is None:
            raise ValueError("You must specify --target with --by-ndvi")
        create_fertilizer_shapefile_by_ndvi(
            args.input, output, args.by_ndvi, args.target
        )


if __name__ == "__main__":
    main()