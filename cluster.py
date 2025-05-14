from abc import ABC, abstractmethod
import rasterio
import geopandas as gpd
import numpy as np
from sklearn.cluster import KMeans
from rasterio.features import shapes
from shapely.geometry import shape
from scipy.ndimage import label
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from clusterable import NDVIData
from common import clusters_info

WORKERS = 4


class BaseRasterClustering(ABC):

    @classmethod
    def fit(cls, data, ndvi: NDVIData) -> None:
        pass

    @staticmethod
    def export_shapefile(data, output_shp: str, transform, crs) -> None:
        pass

    @staticmethod
    def _process_block(block: np.ndarray, min_area: float, pixel_width: float, pixel_height: float) -> np.ndarray:
        labeled_array, _ = label(block)
        pixel_area_m2 = pixel_width * pixel_height  # Размер одного пикселя в м²
        min_area_pixels = int(min_area / pixel_area_m2)

        # print(min_area_pixels)

        # Подсчёт площади каждого региона (кол-во пикселей с каждым label)
        counts = np.bincount(labeled_array.ravel())

        # Массив маски — True для маленьких регионов
        small_regions = np.isin(labeled_array, np.where(counts < min_area_pixels)[0])

        # Удаляем маленькие регионы
        new_block = block.copy()
        new_block[small_regions] = np.nan

        return new_block

    @classmethod
    def _process_block_wrapper(cls, args):
        block, min_area, pixel_width, pixel_height = args
        return cls._process_block(block, min_area, pixel_width, pixel_height)

    @classmethod
    def _clean_small_clusters(cls, clustered, pixel_width, pixel_height, min_area: float = 50, block_size=(2000, 2000)) -> np.ndarray:
        # Определение размера блока (например, 1000x1000 пикселей)
        _, n_rows, n_cols = clustered.shape
        processed = np.zeros_like(clustered)

        blocks = []
        for i in range(0, n_rows, block_size[0]):
            for j in range(0, n_cols, block_size[1]):
                block = clustered[i:i+block_size[0], j:j+block_size[1]]
                blocks.append((block, min_area, pixel_width, pixel_height))

        # Используем многозадачность для параллельной обработки блоков
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            results = list(tqdm(executor.map(cls._process_block_wrapper, blocks), total=len(blocks), desc="Processing blocks"))

        # Объединяем обработанные блоки в один массив
        idx = 0
        for i in range(0, n_rows, block_size[0]):
            for j in range(0, n_cols, block_size[1]):
                processed[i:i+block_size[0], j:j+block_size[1]] = results[idx]
                idx += 1

        return processed


class KMeansRasterClustering(BaseRasterClustering):

    @classmethod
    def fit(cls, ndvi: NDVIData, n_clusters: int, min_cluster_area: float = 50) -> np.ndarray:
        flat = ndvi.data.flatten()
        mask = ~np.isnan(flat)
        valid_data = flat[mask].reshape(-1, 1)

        print(f"Running KMeans with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(valid_data)

        clustered = np.full(flat.shape, np.nan)
        clustered[mask] = kmeans.labels_
        result = clustered.reshape(ndvi.data.shape)
        print("Clustering done.")

        unique = np.unique(result[~np.isnan(result)])
        clusters_info(result, unique, flat)

        if min_cluster_area:
            result = cls._clean_small_clusters(result, ndvi.pixel_width, ndvi.pixel_height, min_cluster_area)
            print("Clean small clusters done.")
            print("Осталось уникальных значений:", np.unique(result[~np.isnan(result)]))

        return result

    @staticmethod
    def export_shapefile(data, output_shp: str, transform, crs) -> None:
        mask = ~np.isnan(data)
        prepared_data = np.nan_to_num(data, nan=-1).astype(np.int16)

        shape_gen = shapes(prepared_data, mask=mask, transform=transform)
        shapes_list = [(shape(g), int(v)) for g, v in shape_gen]
        if not shapes_list:
            raise ValueError("Нет доступных кластеров для экспорта (все удалены?)")

        geoms, ids = zip(*shapes_list)

        gdf = gpd.GeoDataFrame({'cluster_id': ids, 'geometry': geoms}, crs=crs)
        gdf.to_file(output_shp)
        print(f"Shapefile saved to {output_shp}")

