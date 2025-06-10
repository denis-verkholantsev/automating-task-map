# NDVI Processing Toolkit

Набор модулей для обработки многоспектральных изображений, расчёта NDVI, обрезки по границе, кластеризации, разбиения значений по сетке и распределения удобрений.

## Установка

```bash
pip install -r requirements.txt
```
**N.B.** работоспособность скриптов гарантируется с python3.10

## Модули и команды

### 1. NDVI-преобразователь

Преобразует многоспектральное изображение (GeoTIFF) с каналами RED и NIR в изображение NDVI.

**Параметры:**
- `input` — путь к входному GeoTIFF.
- `--output`, `-o` — путь к выходному NDVI GeoTIFF.
- `--red-band` — индекс канала RED (по умолчанию: `3`).
- `--nir-band` — индекс канала NIR (по умолчанию: `5`).

**Пример запуска:**
```bash
python ndvi_converter.py input.tif output_ndvi.tif --red-band 3 --nir-band 5
```

---

### 2. Модуль обрезки по границе

Обрезает NDVI-растр по векторному файлу границ.

**Параметры:**
- `input` — путь к NDVI GeoTIFF (обязательный).
- `boundary` — путь к векторному файлу (shapefile) (обязательный).
- `--output`, `-o` — путь к выходному GeoTIFF.
- `--encoding`, `-e` — кодировка векторного файла (по умолчанию: `latin1`).

**Пример запуска:**
```bash
python raster_clipper.py ndvi.tif boundary.shp --output clipped_ndvi.tif --encoding utf-8
```

---

### 3. Модуль кластеризации

Выполняет кластеризацию NDVI изображения.

**Параметры:**
- `input` — путь к NDVI GeoTIFF.
- `--output`, `-o` — путь к выходному Shapefile.
- `--clusters_number`, `-k` — количество кластеров.
- `--min_area` — минимальная площадь кластера (в м²).
- `--compress` — коэффициенты сжатия изображения (например: `1.5 1.5`).
- `--work_block` — размер рабочего блока (например: `10 10`).
- `--postprocessing_method` — метод агрегации мелких кластеров:
  - `most_common_label`
  - `bfs_nearest`
  - `bfs_most_common`
  - `label_nearest`
  - `label_most_common`
- `--envfile` — путь к файлу переменных окружения.

**Пример запуска:**
```bash
python cluster.py input_ndvi.tif -o result.shp -k 5 --min_area 8 --compress 1.5 1.5 --work_block 10 10 --postprocessing_method most_common_label
```

---

### 4. Модуль разбиения сеткой

Делит NDVI-изображение на блоки и рассчитывает средние значения по ним.

**Параметры:**
- `input` — путь к NDVI GeoTIFF.
- `--output`, `-o` — путь к выходному Shapefile.
- `--block` — размеры блока в метрах: `высота ширина`.

**Пример запуска:**
```bash
python grid.py input_ndvi.tif --block 100 100 -o grid.shp
```

---

### 5. Модуль распределения удобрений

Назначает дозы удобрений по кластерам или сетке.

**Параметры:**
- `input` — путь к входному Shapefile с колонками.
- `--output`, `-o` — путь к выходному Shapefile.
- `--manual` — вручную указать дозу для каждого `cluster_id`, например: `1=50 2=60 3=70`.
- `--by-cluster-id` — рассчитать дозу по NDVI заданного `cluster_id`.
- `--by-ndvi` — рассчитать дозу по NDVI объекта.
- `--target` — целевая доза (обязательно при `--by-cluster-id` или `--by-ndvi`).

**Примеры запуска:**
```bash
python fertilizer.py input.shp --manual 1=50 2=60 3=70 -o result.shp
python fertilizer.py input.shp --by-cluster-id 2 --target 100 -o result.shp
python fertilizer.py input.shp --by-ndvi 2 --target 90 -o result.shp
```
