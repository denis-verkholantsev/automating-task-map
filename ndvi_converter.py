import numpy as np
import rasterio
import argparse

SATURATION_VALUES = {
    'uint8': 255,
    'uint16': 65535,
    'int16': 32767,
    'uint32': 4294967295,
    'float32': None,
    'float64': None
}

def calculate_ndvi_and_save(input_path, output_path, red_band_index=2, nir_band_index=5):
    with rasterio.open(input_path) as src:
        dtype_red = src.dtypes[red_band_index - 1]
        dtype_nir = src.dtypes[nir_band_index - 1]
        src_nodata = src.nodata

        no_data_value = src_nodata if src_nodata is not None else None

        red = src.read(red_band_index).astype(np.float32)
        nir = src.read(nir_band_index).astype(np.float32)

        saturation_red = SATURATION_VALUES.get(dtype_red, None)
        saturation_nir = SATURATION_VALUES.get(dtype_nir, None)

        saturation_mask_red = np.zeros_like(red, dtype=bool)
        saturation_mask_nir = np.zeros_like(nir, dtype=bool)

        if saturation_red is not None:
            saturation_mask_red = (red == saturation_red)
        if saturation_nir is not None:
            saturation_mask_nir = (nir == saturation_nir)

        # маска насыщения, если хотя бы в одном канале есть saturation
        saturation_mask = saturation_mask_red | saturation_mask_nir

        # маска для nodata, где есть saturation
        if src_nodata is not None:
            saturation_mask |= (red == src_nodata) | (nir == src_nodata)

        ndvi = (nir - red) / (nir + red + 1e-12)
        ndvi[saturation_mask] = no_data_value

        out_meta = src.meta.copy()
        out_meta.update({
            "count": 1,
            "dtype": "float32",
            "nodata": no_data_value
        })

        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(ndvi, 1)


def main():
    parser = argparse.ArgumentParser(description='Calculate NDVI from multispectral image')
    parser.add_argument('input', help='Input multispectral raster file')
    parser.add_argument('output', help='Output NDVI raster file')
    parser.add_argument('--red-band', type=int, default=3, help='Band index for Red channel (default: 3)')
    parser.add_argument('--nir-band', type=int, default=5, help='Band index for NIR channel (default: 5)')

    args = parser.parse_args()

    calculate_ndvi_and_save(
        input_path=args.input,
        output_path=args.output,
        red_band_index=args.red_band,
        nir_band_index=args.nir_band,
    )

if __name__ == '__main__':
    main()


# # Пример использования
# calculate_ndvi_and_save(
#     input_path='/home/dverholancev/study/degree/experience_2023/20230619_F14_Micasense.tif',
#     output_path='ndvi_output.tif',
#     red_band_index=3,
#     nir_band_index=5
# )
