import numpy as np
from numpy.lib.stride_tricks import as_strided
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
from clusterable import NDVIData


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
    means = np.nanmean(blocks, axis=(2, 3))  # игнорируем NaN, считаем среднее

    return means.repeat(block_h, axis=0).repeat(block_w, axis=1)


def block_average_with_edges(ndvi: NDVIData, block_size: tuple[float, float]) -> np.ndarray:
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
    gdf = gpd.GeoDataFrame({'means': means, 'geometry': geoms}, crs=crs)
    gdf.to_file(output_shp)
    print(f"Shapefile saved to {output_shp}")
