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

    def fit(ndvi, n_clusters: int, min_cluster_area: float = 10):
        flat = ndvi.data.flatten()
        mask = ~np.isnan(flat)
        valid_data = flat[mask].reshape(-1, 1)

        print(f"Running KMeans with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(valid_data)

        clustered = np.full(flat.shape, np.nan)
        clustered[mask] = kmeans.labels_
        result = clustered.reshape(ndvi.data.shape)

        print("Clustering done.")
        
        if min_cluster_area == 0:
            return result
        
        cleaned = KMeansRasterClustering._clean_small_clusters(result, ndvi.pixel_width, ndvi.pixel_height, min_area_m2=50) 
        print("Clean small clusters done.")

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
        geoms, ids = zip(*[(shape(g), int(v)) for g, v in shape_gen])

        gdf = gpd.GeoDataFrame({'cluster_id': ids, 'geometry': geoms}, crs=crs)
        gdf.to_file(output_shp)
        print(f"Shapefile saved to {output_shp}")

    def _clean_small_clusters_worker(cluster_id, clustered, pixel_width, pixel_height, min_area_m2):
        pixel_area_m2 = pixel_width * pixel_height
        mask = clustered == cluster_id

        area_pixels = np.sum(mask)
        area_m2 = area_pixels * pixel_area_m2

        if area_m2 < min_area_m2:
            return mask
        else:
            return None

    def _clean_small_clusters(clustered, pixel_width, pixel_height, min_area_m2=50, n_jobs=4):
        clustered = np.copy(clustered)
        unique_clusters = np.unique(clustered[~np.isnan(clustered)])

        small_cluster_masks = Parallel(n_jobs=n_jobs)(
            delayed(KMeansRasterClustering._clean_small_clusters_worker)(cluster_id, clustered, pixel_width, pixel_height, min_area_m2)
            for cluster_id in tqdm(unique_clusters, desc="Cleaning small clusters")
        )

        for mask in small_cluster_masks:
            if mask is not None:
                clustered[mask] = np.nan

        return clustered