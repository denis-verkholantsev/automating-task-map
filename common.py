import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.ops import polygonize
from pathlib import Path
import math


GEOM_TYPES_TO_RETURN = {"Polygon", 'MultiPolygon'}


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


def analyze_ndvi(ndvi_array):
    valid = ndvi_array[~np.isnan(ndvi_array)]

    if valid.size == 0:
        print("Нет валидных NDVI значений.")
        return

    print(f"Количество пикселей: {valid.size}")
    print(f"Минимум: {valid.min():.4f}")
    print(f"Максимум: {valid.max():.4f}")
    print(f"Среднее: {valid.mean():.4f}")
    print(f"Медиана: {np.median(valid):.4f}")
    print(f"Уникальных значений: {np.unique(valid).size}")

    plt.hist(valid, bins=50, color='green', edgecolor='black')
    plt.title("Распределение NDVI")
    plt.xlabel("NDVI значение")
    plt.ylabel("Частота")
    plt.grid(True)
    plt.show()


def get_ndvi_stats_by_ranges(ndvi_data, step=0.05):
    mask = (ndvi_data >= -1) & (ndvi_data <= 1)
    ndvi_data = ndvi_data[mask]

    ranges = np.arange(-1, 1 + step, step)

    print(f"Статистика по диапазонам значений NDVI (в диапазоне от -1 до 1):")
    
    for i in range(len(ranges) - 1):
        lower = ranges[i]
        upper = ranges[i + 1]
        
        range_mask = (ndvi_data >= lower) & (ndvi_data < upper)
        range_vals = ndvi_data[range_mask]
        
        if range_vals.size > 0:
            print(f"\nДиапазон: {lower:.2f} - {upper:.2f}")
            print(f"Количество пикселей: {range_vals.size}")
            print(f"Минимум: {np.min(range_vals)}")
            print(f"Максимум: {np.max(range_vals)}")
            print(f"Среднее: {np.mean(range_vals)}")
            print(f"Медиана: {np.median(range_vals)}")
        else:
            print(f"\nДиапазон: {lower:.2f} - {upper:.2f} - Нет данных")


from pyproj import CRS, Transformer

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
    print("Информация о кластерах:")
    for val in unique:
        cluster_pixels = flat[result.flatten() == val]  # Пиксели, принадлежащие текущему кластеру
        min_value = np.min(cluster_pixels)
        max_value = np.max(cluster_pixels)
        print(f"Кластер {val}: min = {min_value}, max = {max_value}")


def get_block_size(min_area_m2, pixel_width_m, pixel_height_m):
    block_area_m2 = min_area_m2 / 0.33
    block_side_m = block_area_m2 ** 0.5
    width_px = math.ceil(block_side_m / pixel_width_m)
    height_px = math.ceil(block_side_m / pixel_height_m)
    return height_px, width_px
