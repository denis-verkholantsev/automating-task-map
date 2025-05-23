import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.ops import polygonize
from pathlib import Path
from pyproj import CRS, Transformer
import time


GEOM_TYPES_TO_RETURN = {"Polygon", 'MultiPolygon'}
MIN_AREA_BLOCK_PART = 3


def make_polygon_filepath(path: str) -> str:
    p = Path(path)
    return str(p.with_name(p.stem + '_polygon.shp'))


def lines_to_polygon(input_lines_shp: str, encoding: str | None = 'utf-8') -> str:
    gdf = gpd.read_file(input_lines_shp, encoding=encoding)

    all_lines = []
    for geom in gdf.geometry:
        if geom.geom_type in GEOM_TYPES_TO_RETURN:
            return input_lines_shp
        elif geom.geom_type == "LineString":
            all_lines.append(geom)
        elif geom.geom_type == "MultiLineString":
            all_lines.extend(list(geom.geoms))
        
    polygons = list(polygonize(all_lines))

    poly_gdf = gpd.GeoDataFrame(geometry=polygons, crs=gdf.crs)
    output_polygon_shp = make_polygon_filepath(input_lines_shp)
    poly_gdf.to_file(output_polygon_shp)

    return output_polygon_shp


def convert_pixel_size_to_meters(pixel_width_deg, pixel_height_deg, center_lon, center_lat):
    crs_src = CRS("EPSG:4326")
    utm_zone = int((center_lon + 180) / 6) + 1
    is_northern = center_lat >= 0
    epsg_code = 32600 + utm_zone if is_northern else 32700 + utm_zone
    crs_dst = CRS.from_epsg(epsg_code)

    transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)

    x0, y0 = transformer.transform(center_lon, center_lat)
    x1, _ = transformer.transform(center_lon + pixel_width_deg, center_lat)
    _, y1 = transformer.transform(center_lon, center_lat - pixel_height_deg)

    pixel_width_m = abs(x1 - x0)
    pixel_height_m = abs(y1 - y0)

    return pixel_width_m, pixel_height_m


def clusters_info(result, unique, flat):
    print("Clusters info:")
    stats = {}
    for val in unique:
        cluster_pixels = flat[result.flatten() == val]  # Пиксели, принадлежащие текущему кластеру
        min_value = np.min(cluster_pixels)
        max_value = np.max(cluster_pixels)
        mean_value = np.mean(cluster_pixels)
        stats[val] = {'min': min_value, 'max': max_value, 'mean': mean_value}
        print(f"Сluster {val}: min = {min_value}, max = {max_value}, mean = {mean_value}")

    return stats


def get_block_size(block_size: tuple[float, float], min_area_m2: float,
                   pixel_width_m: float, pixel_height_m: float) -> tuple[float, float]:
    if block_size is None:
        block_area_m2 = min_area_m2 / (1 / MIN_AREA_BLOCK_PART)
        block_side_m = block_area_m2 ** 0.5
        block_size = (block_side_m, block_side_m)

    return get_block_size_from_meters_to_px(block_size, pixel_width_m, pixel_height_m)


def get_block_size_from_meters_to_px(block_size: tuple[float, float],
                                     pixel_width_m: float, pixel_height_m: float) -> tuple[float, float]:
    return int(block_size[0] // pixel_height_m), int(block_size[1] // pixel_width_m)


def decompress_array(data: np.ndarray, block_size: tuple[int, int]) -> np.ndarray:
    return data.repeat(block_size[0], axis=0).repeat(block_size[1], axis=1)


def create_filepath(path: str, ext: str | None = None, *args):
    input_path = Path(path)
    now_ts = int(time.time())
    parts = [str(arg) for arg in args if arg]
    parts.append(str(now_ts))
    result = "_".join(parts)
    ext = ext or input_path.suffix or ''

    return input_path.with_name(f"{input_path.stem}_{result}.{ext}")