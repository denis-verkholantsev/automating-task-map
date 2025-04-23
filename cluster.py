from abc import ABC, abstractmethod
import rasterio
import geopandas as gpd
import numpy as np
from sklearn.cluster import KMeans
from rasterio.features import shapes
from shapely.geometry import shape
from scipy.ndimage import label
from skimage.measure import regionprops
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor

from clusterable import NDVIData


class BaseRasterClustering(ABC):

    @abstractmethod
    def fit(data, ndvi: NDVIData) -> None:
        pass
    
    @abstractmethod
    def save_clustered_raster(data, output_path: str, profile: dict) -> None:
        pass

    @abstractmethod
    def export_shapefile(data, output_shp: str, transform, crs) -> None:
        pass


class KMeansRasterClustering(BaseRasterClustering):

    def fit(ndvi: NDVIData, n_clusters: int, min_cluster_area: float = 10, min_area=50):
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
        print("Информация о кластерах:")
        for val in unique:
            cluster_pixels = flat[result.flatten() == val]  # Пиксели, принадлежащие текущему кластеру
            min_value = np.min(cluster_pixels)
            max_value = np.max(cluster_pixels)
            print(f"Кластер {val}: min = {min_value}, max = {max_value}")
        
        if min_cluster_area == 0:
            return result
        
        cleaned = KMeansRasterClustering._clean_small_clusters(result, ndvi.pixel_width, ndvi.pixel_height, min_area) 
        print("Clean small clusters done.")
        print("Осталось уникальных значений:", np.unique(result[~np.isnan(result)]))

        return cleaned

    def save_clustered_raster(data, output_path: str, profile: dict) -> None:
        profile = profile.copy()
        profile.update(dtype=rasterio.uint8, count=1, nodata=0)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data.astype(rasterio.uint8))
        print(f"Clustered raster saved to {output_path}")

    def export_shapefile(data,output_shp: str, transform, crs) -> None:

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

    def process_block(block, min_area, pixel_width, pixel_height):
        labeled_array, _ = label(block)
        pixel_area_m2 = pixel_width * pixel_height  # Размер одного пикселя в м²
        min_area_pixels = int(min_area / pixel_area_m2)
        
        # Подсчёт площади каждого региона (кол-во пикселей с каждым label)
        counts = np.bincount(labeled_array.ravel())

        # Массив маски — True для маленьких регионов
        small_regions = np.isin(labeled_array, np.where(counts < min_area_pixels)[0])

        # Удаляем маленькие регионы
        new_block = block.copy()
        new_block[small_regions] = np.nan

        return new_block

    def process_block_wrapper(args):
        block, min_area, pixel_width, pixel_height = args
        return KMeansRasterClustering.process_block(block, min_area, pixel_width, pixel_height)

    def _clean_small_clusters(clustered, pixel_width, pixel_height, min_area=100, block_size=(2000, 2000)):
        # Определение размера блока (например, 1000x1000 пикселей)
        _, n_rows, n_cols = clustered.shape
        processed = np.zeros_like(clustered)
        
        blocks = []
        for i in range(0, n_rows, block_size[0]):
            for j in range(0, n_cols, block_size[1]):
                block = clustered[i:i+block_size[0], j:j+block_size[1]]
                blocks.append((block, min_area, pixel_width, pixel_height))
        
        # Используем многозадачность для параллельной обработки блоков
        with ProcessPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(executor.map(KMeansRasterClustering.process_block_wrapper, blocks), total=len(blocks), desc="Processing blocks"))

        # Объединяем обработанные блоки в один массив
        idx = 0
        for i in range(0, n_rows, block_size[0]):
            for j in range(0, n_cols, block_size[1]):
                processed[i:i+block_size[0], j:j+block_size[1]] = results[idx]
                idx += 1

        return processed
