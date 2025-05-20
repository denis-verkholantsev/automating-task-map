import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from exceptions import CompressException

from common import convert_pixel_size_to_meters


class NDVIData:
    def __init__(self, path: str | None = None):
        self._path = path
        self._data = None
        self._profile = None
        self._crs = None
        self._pixel_area = None
        self._pixel_width = None
        self._pixel_height = None
        self._compressed = None
    
    @staticmethod
    def from_data(image, meta, crs, pixel_width, pixel_height, path=None) -> 'NDVIData':
        result = NDVIData(path)
        result._data = image
        result._profile = meta
        result._profile["crs"] = crs
        result._crs = crs
        result._pixel_width = pixel_width
        result._pixel_height = pixel_height
        result._compressed = None

        return result
    
    @staticmethod
    def load(path: str) -> 'NDVIData':
        result = NDVIData(path)
        with rasterio.open(path) as src:
            result._data = src.read()
            result._profile = src.profile
            result._crs = src.crs
            result._pixel_width, result._pixel_height = src.res
            result._pixel_area = abs(result._pixel_width * result._pixel_height)

        if result._crs.to_epsg() == 4326:
            center_lon = (src.bounds.left + src.bounds.right) / 2
            center_lat = (src.bounds.top + src.bounds.bottom) / 2

            pixel_width_m, pixel_height_m = convert_pixel_size_to_meters(
                result._pixel_width, result._pixel_height,
                center_lon, center_lat
            )

            result._pixel_width = pixel_width_m
            result._pixel_height = pixel_height_m
            result._pixel_area = pixel_width_m * pixel_height_m
        else:
            result._pixel_area = abs(result._pixel_width * result._pixel_height)

        result._compressed = None

        print(f"Converted pixel size to meters: {result._pixel_width} x {result._pixel_height} meters.")
        print(f"NDVI loaded from {result._path}")

        return result

    def clean(self):
        self._data = np.where((self._data <= 0) | (self._data > 1), np.nan, self._data)
        print("NDVI cleaned")

    def save(self, path: str):
        if self._data is None or self._profile is None:
            raise ValueError("NDVI data or profile not set.")

        with rasterio.open(path, 'w', **self._profile) as dst:
            dst.write(self._data)
        print(f"NDVI saved to {path}")

    def compress(self, block_size: tuple[float, float], pixel: bool=False):
        """
        Сжать 0-й канал усреднением блоков block_size x block_size.
        Изменит _data, _profile, _pixel_width, _pixel_height.
        """

        if not pixel:
            block_size = (int(block_size[0] // self.pixel_height), int(block_size[1] // self.pixel_width))

        band_data = self._data[0]
        height, width = band_data.shape

        # Вычисляем размер с паддингом
        pad_height = (block_size[0] - height % block_size[0]) % block_size[0]
        pad_width = (block_size[1] - width % block_size[1]) % block_size[1]

        # Паддинг снизу и справа
        padded = np.pad(band_data, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=np.nan)

        new_height = (height + pad_height) // block_size[0]
        new_width = (width + pad_width) // block_size[1]

        # Переформатируем для блочного усреднения
        reshaped = padded.reshape(new_height, block_size[0], new_width, block_size[1])

        # Усреднение по блокам
        compressed_band = np.nanmean(reshaped, axis=(1, 3))

        # Обновляем _data, оставляя только один канал
        self._data = compressed_band[None, :, :]  # добавляем размерность для bands=1

        # Обновляем размеры пикселя
        self._pixel_height *= block_size[0]
        self._pixel_width *= block_size[1]
        self._pixel_area = abs(self._pixel_width * self._pixel_height)

        # Обновляем профиль
        transform = self._profile['transform']
        new_transform = Affine(
            transform.a * block_size[1], transform.b, transform.c,
            transform.d, transform.e * block_size[0], transform.f
        )
        self._profile['transform'] = new_transform
        self._profile['height'] = new_height
        self._profile['width'] = new_width
        self._compression_block = block_size
        self._compressed = True

        print(f"Compressed 0-th band to shape {self._data.shape} with pixel size {self._pixel_width} x {self._pixel_height} m")

    def decompress(self):
        """
        Разжать данные без интерполяции.
        Каждый пиксель становится прямоугольником H x W с одинаковыми значениями.
        block_size — кортеж (H, W), с которым ранее сжималось.
        """
        if not self.is_compressed():
            raise CompressException("Data is not compressed")

        if self._data is None:
            raise ValueError("No data to decompress")

        H, W = self.compression_block

        # Повторяем каждый пиксель
        decompressed = self._data.repeat(H, axis=1).repeat(W, axis=2)

        self._data = decompressed

        # Обновляем размеры пикселя
        self._pixel_width /= W
        self._pixel_height /= H
        self._pixel_area = abs(self._pixel_width * self._pixel_height)

        # Обновляем профиль
        transform = self._profile['transform']
        new_transform = Affine(
            transform.a / W, transform.b, transform.c,
            transform.d, transform.e / H, transform.f
        )
        self._profile['transform'] = new_transform
        self._profile['height'] = decompressed.shape[1]
        self._profile['width'] = decompressed.shape[2]

        self._compressed = False

        print(f"Decompressed data to shape {self._data.shape} with pixel size {self._pixel_width} x {self._pixel_height} m")

        return self._compression_block


    @property
    def data(self):
        return self._data
    
    @property
    def crs(self):
        return self._crs
    
    @property
    def path(self):
        return self._path
    
    @property
    def profile(self):
        return self._profile
    
    @property
    def transform(self):
        return self._profile.get('transform')
    
    @property
    def pixel_width(self):
        return self._pixel_width
    
    @property
    def pixel_height(self):
        return self._pixel_height

    def is_compressed(self):
        return self._compressed

    @property
    def compression_block(self):
        if self._compressed:
            return self._compression_block

    def size(self):
        if self._data:
            return self._data.shape[1] * self._data.shape[2]
