from abc import ABC
import geopandas as gpd
import numpy as np
import argparse
from sklearn.cluster import KMeans, MiniBatchKMeans
from rasterio.features import shapes
from shapely.geometry import shape
from scipy.ndimage import label, generate_binary_structure, binary_dilation
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from functools import partial

from clusterable import NDVIData
from common import clusters_info, get_block_size, create_filepath, decompress_array
from exceptions import ArgumentException
from config import Config

WORKERS = 4
MAX_SIZE_DEFAULT_KMEANS = 50000000

class BaseRasterClustering(ABC):

    PHRASE_TO_CLEAN_METHOD = {
        'most_common_label': '_process_block_with_most_common_label',
        'bfs_most_common': '_process_block_bfs_with_most_common_label',
        'bfs_nearest': '_process_block_bfs_with_nearest_label',
        'label_most_common': '_process_block_label_with_most_common_label',
        'label_nearest': '_process_block_label_with_nearest_label',
    }

    @classmethod
    def fit(cls, data, ndvi: NDVIData) -> None:
        pass

    @staticmethod
    def export_shapefile(data, output_shp: str, transform, crs, stats: dict=None) -> gpd.GeoDataFrame:
        # в 2d
        data_2d = data.squeeze()
        mask = ~np.isnan(data_2d)
        prepared_data = np.nan_to_num(data_2d, nan=-1).astype(np.int16)

        shape_gen = shapes(prepared_data, mask=mask, transform=transform)
        shapes_list = [(shape(g), int(v)) for g, v in shape_gen]

        if not shapes_list:
            raise ValueError("No clusters available for export (all removed?)")

        geoms, ids = zip(*shapes_list)
        data_to_gdf = {'cluster_id': ids, 'geometry': geoms}
        if stats:
            means = [stats[cluster_id]['mean'] for cluster_id in ids]
            data_to_gdf['mean'] = means
        gdf = gpd.GeoDataFrame(data_to_gdf, crs=crs)
        gdf.to_file(output_shp)
        print(f"Shapefile saved to {output_shp}")

        return gdf

    @staticmethod
    def _process_block_with_most_common_label(block: np.ndarray, *args) -> np.ndarray:
        processed = block.copy()
        nan_mask = np.isnan(processed)
        processed[nan_mask] = -1  # временная метка для NaN

        # наиболее популярное значение (кластер) в блоке
        valid_labels = processed[processed > 0]
        most_common_label = np.bincount(valid_labels.astype(int)).argmax() if len(valid_labels) > 0 else -1

        if most_common_label > 0:
            processed[:, :] = most_common_label
        else:
            processed[:, :] = np.nan  # если нет валидных значений

        processed[processed == -1] = np.nan
        return processed


    @staticmethod
    def _process_block_bfs_with_most_common_label(block: np.ndarray, min_area: float, pixel_width: float, pixel_height: float, connectivity = 4) -> np.ndarray:
        min_pixels = max(1, int(min_area / (pixel_width * pixel_height)))
        processed = block.copy()
        nan_mask = np.isnan(processed)
        processed[nan_mask] = -1  # временная метка для NaN

        h, w = processed.shape
        visited = np.zeros((h, w), dtype=bool)

        # Наиболее частая метка
        valid_labels = processed[processed > 0]
        most_common_label = np.bincount(valid_labels.astype(int)).argmax() if len(valid_labels) > 0 else -1

        directions_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        directions_8 = directions_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        directions = directions_4 if connectivity == 4 else directions_8

        def bfs(start_y, start_x, label_value):
            queue = deque()
            queue.append((start_y, start_x))
            region = [(start_y, start_x)]
            visited[start_y, start_x] = True

            while queue:
                y, x = queue.popleft()
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if not visited[ny, nx] and processed[ny, nx] == label_value:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
                            region.append((ny, nx))
            return region

        for y in range(h):
            for x in range(w):
                if visited[y, x]:
                    continue
                label_value = processed[y, x]
                if label_value <= 0:
                    continue

                region = bfs(y, x, label_value)

                if len(region) < min_pixels and most_common_label > 0 and most_common_label != label_value:
                    for ry, rx in region:
                        processed[ry, rx] = most_common_label

        processed[processed == -1] = np.nan
        return processed

    @staticmethod
    def _process_block_bfs_with_nearest_label(block: np.ndarray, min_area: float, pixel_width: float, pixel_height: float, connectivity = 4) -> np.ndarray:
        min_pixels = max(1, int(min_area / (pixel_width * pixel_height)))
        processed = block.copy()
        nan_mask = np.isnan(processed)
        processed[nan_mask] = -1  # NaN

        h, w = processed.shape
        visited = np.zeros((h, w), dtype=bool)

        directions_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        directions_8 = directions_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        directions = directions_4 if connectivity == 4 else directions_8

        def bfs(start_y, start_x, label_value):
            queue = deque()
            queue.append((start_y, start_x))
            region = [(start_y, start_x)]
            visited[start_y, start_x] = True

            while queue:
                y, x = queue.popleft()
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if not visited[ny, nx] and processed[ny, nx] == label_value:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
                            region.append((ny, nx))
            return region

        for y in range(h):
            for x in range(w):
                if visited[y, x]:
                    continue
                label_value = processed[y, x]
                if label_value <= 0:
                    continue

                region = bfs(y, x, label_value)

                from collections import Counter

                if len(region) < min_pixels:
                    neighbor_labels = []

                    for ry, rx in region:
                        for dy, dx in directions:
                            ny, nx = ry + dy, rx + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                neighbor_label = processed[ny, nx]
                                if neighbor_label > 0 and neighbor_label != label_value:
                                    neighbor_labels.append(neighbor_label)

                    if neighbor_labels:
                        most_common_neighbor = Counter(neighbor_labels).most_common(1)[0][0]
                        for ry, rx in region:
                            processed[ry, rx] = most_common_neighbor

        processed[processed == -1] = np.nan
        return processed

    @staticmethod
    def _process_block_label_with_most_common_label(block: np.ndarray, min_area: float, pixel_width: float, pixel_height: float, connectivity = 4) -> np.ndarray:
        min_pixels = max(1, int(min_area / (pixel_width * pixel_height)))
        processed = block.copy()
        nan_mask = np.isnan(processed)
        processed[nan_mask] = -1  # NaN

        structure = None
        if connectivity == 4:
            structure = generate_binary_structure(2, 1)  # связность 1 (четыре соседа)
        elif connectivity == 8:
            structure = generate_binary_structure(2, 2)  # связность 2 (восемь соседей)

        # наиболее популярное значение в блоке
        valid_labels = processed[processed > 0]
        most_common_label = np.bincount(valid_labels.astype(int)).argmax() if len(valid_labels) > 0 else -1

        # Обрабатываем каждое уникальное значение отдельно (каждый кластер)
        for label_value in np.unique(processed):
            if label_value <= 0:
                continue  # Пропускаем фон и NaN

            # Маска только данного кластера
            mask = processed == label_value

            # Находим связные компоненты внутри кластера
            labeled, num_features = label(mask, structure)

            for region_id in range(1, num_features + 1):
                region_mask = labeled == region_id
                region_area = np.sum(region_mask)

                if region_area < min_pixels and most_common_label > 0 and most_common_label != label_value:
                    processed[region_mask] = most_common_label

        processed[processed == -1] = np.nan
        return processed

    @staticmethod
    def _process_block_label_with_nearest_label(block: np.ndarray, min_area: float, pixel_width: float, pixel_height: float, connectivity = 4) -> np.ndarray:
        min_pixels = max(1, int(min_area / (pixel_width * pixel_height)))
        processed = block.copy()
        nan_mask = np.isnan(processed)
        processed[nan_mask] = -1  # Временная метка для NaN

        structure = None
        if connectivity == 4:
            structure = generate_binary_structure(2, 1)  # 2D, связность 1 (четыре соседа)
        elif connectivity == 8:
            structure = generate_binary_structure(2, 2)  # 2D, связность 2 (восемь соседей)

        # Обрабатываем каждое уникальное значение отдельно (каждый кластер)
        for label_value in np.unique(processed):
            if label_value <= 0:
                continue  # Пропускаем фон и NaN

            # Маска только данного кластера
            mask = processed == label_value

            # Находим связные компоненты внутри кластера
            labeled, num_features = label(mask, structure)

            for region_id in range(1, num_features + 1):
                region_mask = labeled == region_id
                region_area = np.sum(region_mask)

                if region_area >= min_pixels:
                    continue

                # Получим координаты граничных пикселей
                border = binary_dilation(region_mask, structure) & ~region_mask
                neighbor_labels = processed[border]
                neighbor_labels = neighbor_labels[(neighbor_labels > 0) & (neighbor_labels != label_value)]

                if neighbor_labels.size > 0:
                    new_label = np.bincount(neighbor_labels.astype(int)).argmax()
                    processed[region_mask] = new_label

        processed[processed == -1] = np.nan
        return processed

    @classmethod
    def get_clean_method(cls, phrase: str):
        method_name = cls.PHRASE_TO_CLEAN_METHOD[phrase]
        return getattr(cls, method_name)

    @staticmethod
    def _process_block_wrapper(args, clean_method):
        block, min_area, pixel_width, pixel_height = args
        return clean_method(block, min_area, pixel_width, pixel_height)

    @classmethod
    def _clean_small_clusters(cls, clustered, pixel_width, pixel_height,
                              min_area: float = 50, block_size=(50, 50), clean_method=None) -> np.ndarray:
        n_rows, n_cols = clustered.shape
        processed = np.zeros_like(clustered)

        # разбиваем на блоки
        blocks = []
        for i in range(0, n_rows, block_size[0]):
            for j in range(0, n_cols, block_size[1]):
                block = clustered[i:i+block_size[0], j:j+block_size[1]]
                blocks.append((block, min_area, pixel_width, pixel_height))

        clean_func = cls.get_clean_method(clean_method)  # или любой другой
        wrapper = partial(cls._process_block_wrapper, clean_method=clean_func)

        # параллельно обрабатываем
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            results = list(tqdm(
                executor.map(wrapper, blocks),
                total=len(blocks),
                desc="Processing blocks"
            ))

        # пересобираем обработанные
        idx = 0
        for i in range(0, n_rows, block_size[0]):
            for j in range(0, n_cols, block_size[1]):
                processed[i:i+block_size[0], j:j+block_size[1]] = results[idx]
                idx += 1

        return processed


class KMeansRasterClustering(BaseRasterClustering):
    @classmethod
    def fit(cls,
            ndvi: NDVIData,
            n_clusters: int | None = 3,
            min_cluster_area: float | None = 50,
            clean_method: str | None = 'most_common_label',
            block_size: tuple[float, float] | None = (1.5, 1.5),
            use_mini_batch: bool | None = False,
            batch_size: int | None = 1000000,
            workers: int | None = WORKERS) -> tuple[np.ndarray, dict]:

        if clean_method == 'most_common_label' and not block_size:
            raise ArgumentException('need block size for most_common_label method')

        # в 1d, избавимся от Nan
        data_2d = ndvi.data.squeeze()  # 2D
        flat = data_2d.flatten()
        mask = ~np.isnan(flat)
        valid_data = flat[mask].reshape(-1, 1)

        # k-means
        print(f"Running KMeans with {n_clusters} clusters...")
        if use_mini_batch:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=batch_size)
            kmeans.fit(valid_data)
            labels = kmeans.predict(valid_data)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(valid_data)
            labels = kmeans.labels_

        # пересобираем и возвращаем nan
        clustered = np.full(flat.shape, np.nan)
        clustered[mask] = labels + 1  # +1
        result = clustered.reshape(data_2d.shape)
        print("Clustering done.")

        # анализ по кластерам
        unique = np.unique(result[~np.isnan(result)])
        stats = clusters_info(result, unique, flat)

        # чистим мелкие области, если они есть
        if min_cluster_area:
            result = cls._clean_small_clusters(
                result,
                ndvi.pixel_width,
                ndvi.pixel_height,
                min_cluster_area,
                block_size=get_block_size(block_size, min_cluster_area, ndvi.pixel_width, ndvi.pixel_height),
                clean_method=clean_method,
            )
            print(f'Clean small clusters done.\nRemaining unique values: {np.unique(result[~np.isnan(result)])}')

        return result, stats


def call(config, args):
    ndvi = NDVIData.load(args.input)
    ndvi.clean()

    to_decompress = False
    if args.compress:
        ndvi.compress(args.compress)
        to_decompress = True

    use_mini_batch = ndvi.size() > config.MAX_SIZE_DEFAULT_KMEANS
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

    if to_decompress:
        block = ndvi.compression_block
        ndvi.decompress()
        result = decompress_array(result, block)

    KMeansRasterClustering.export_shapefile(
        result,
        args.output or create_filepath(args.input, 'shp', 'kmeans'),
        ndvi.transform,
        ndvi.crs,
        stats=stats
    )


def parse_args():
    parser = argparse.ArgumentParser(description="NDVI clustering tool")

    parser.add_argument(
        "input",
        type=str,
        help="path to input-NDVI GeoTIFF"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="path to output-NDVI .shp"
    )

    parser.add_argument(
        "--clusters_number", "-k",
        type=int,
        default=4,
        required=True,
        help="clusters number"
    )

    parser.add_argument(
        "--min_area",
        type=float,
        help="min area in m2"
    )

    parser.add_argument(
        "--compress",
        type=float,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        help="compress size block"
    )

    parser.add_argument(
        "--work_block",
        type=float,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        help="work size block"
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
        help="small clusters aggregation method"
    )

    parser.add_argument(
        "--envfile",
        type=str,
        help="path to .env file"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    call(config, args)


if __name__ == '__main__':
    main()

