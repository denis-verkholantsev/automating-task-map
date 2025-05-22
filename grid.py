import numpy as np
from numpy.lib.stride_tricks import as_strided
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd

from clusterable import NDVIData
from common import create_filepath


def fast_block_average(arr: np.ndarray, block_h: int, block_w: int) -> np.ndarray:
    h, w = arr.shape
    h_trim, w_trim = h - h % block_h, w - w % block_w
    arr_cropped = arr[:h_trim, :w_trim]

    new_shape = (h_trim // block_h, w_trim // block_w, block_h, block_w)
    new_strides = (
        arr.strides[0] * block_h,
        arr.strides[1] * block_w,
        arr.strides[0],
        arr.strides[1]
    )

    blocks = as_strided(arr_cropped, shape=new_shape, strides=new_strides)
    means = np.nanmean(blocks, axis=(2, 3))  # игнорируем nan, считаем среднее

    return means.repeat(block_h, axis=0).repeat(block_w, axis=1)


def block_average_with_edges(ndvi: NDVIData, block_size: tuple[float, float] = (10, 10)) -> np.ndarray:
    block_h = int(block_size[0] // ndvi.pixel_height)
    block_w = int(block_size[1] // ndvi.pixel_width)

    arr = ndvi.data.squeeze()
    h, w = arr.shape
    h_trim, w_trim = h - h % block_h, w - w % block_w

    # основная часть
    fast_part = fast_block_average(arr[:h_trim, :w_trim], block_h, block_w)

    out = np.empty_like(arr, dtype=float)
    out[:h_trim, :w_trim] = fast_part

    # нижняя полоса
    if h_trim < h:
        for j in range(0, w_trim, block_w):
            block = arr[h_trim:, j:min(j + block_w, w)]
            out[h_trim:, j:min(j + block_w, w)] = np.nanmean(block)

    # правая полоса
    if w_trim < w:
        for i in range(0, h_trim, block_h):
            block = arr[i:min(i + block_h, h), w_trim:]
            out[i:min(i + block_h, h), w_trim:] = np.nanmean(block)

    # нижний правый угол
    if h_trim < h and w_trim < w:
        block = arr[h_trim:, w_trim:]
        out[h_trim:, w_trim:] = np.nanmean(block)

    return out


def export_shapefile(data, output_shp: str, transform, crs) -> None:
    data_2d = data.squeeze()
    mask = ~np.isnan(data_2d)
    prepared_data = np.nan_to_num(data_2d, nan=-1).astype(np.float32)

    shape_gen = shapes(prepared_data, mask=mask, transform=transform)
    shapes_list = [(shape(g), float(v)) for g, v in shape_gen]

    if not shapes_list:
        raise ValueError("No values for export (all removed?)")

    geoms, means = zip(*shapes_list)
    gdf = gpd.GeoDataFrame({'mean': means, 'geometry': geoms}, crs=crs)
    gdf.to_file(output_shp)
    print(f"Shapefile saved to {output_shp}")


import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Grid-based mean NDVI")

    parser.add_argument(
        "input",
        type=str,
        help="Путь к входному NDVI GeoTIFF"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Путь к выходному Shapefile (например, output.shp)"
    )

    parser.add_argument(
        "--block",
        type=float,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        help="Размер блока для усреднения в метрах (высота, ширина)"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    ndvi = NDVIData.load(args.input)
    averaged = block_average_with_edges(ndvi, tuple(args.block))
    export_shapefile(averaged, args.output or create_filepath(args.input, 'shp', 'grid'), ndvi.transform, ndvi.crs)


if __name__ == "__main__":
    main()

