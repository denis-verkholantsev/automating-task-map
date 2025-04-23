from osgeo import gdal, ogr, osr

# Открытие растра
raster = gdal.Open("/home/dverholancev/study/degree/src/clipped_by_qgis_mask.tif")
band = raster.GetRasterBand(1)

# Получаем геотрансформ для перевода координат пикселей в географические
gt = raster.GetGeoTransform()

# Размеры растра
rows = band.YSize
cols = band.XSize

# Создание shapefile
driver = ogr.GetDriverByName('ESRI Shapefile')
output_shapefile = driver.CreateDataSource("/home/dverholancev/study/degree/src/clustered.shp")
srs = osr.SpatialReference()
srs.ImportFromWkt(raster.GetProjection())  # Наследуем проекцию растра
layer = output_shapefile.CreateLayer("points", srs, geom_type=ogr.wkbPoint)
layer.CreateField(ogr.FieldDefn("ndvi", ogr.OFTReal))
layer_defn = layer.GetLayerDefn()

# Функция для перевода пиксельных координат в географические
def pixel_to_coords(x, y, gt):
    geo_x = gt[0] + x * gt[1] + y * gt[2]
    geo_y = gt[3] + x * gt[4] + y * gt[5]
    return geo_x, geo_y

# Проходим по строкам и записываем данные
for i in range(rows):
    scanline = band.ReadAsArray(0, i, cols, 1)[0]  # Получаем строку
    for j in range(cols):
        value = scanline[j]
        if value == band.GetNoDataValue():  # Пропускаем пиксели с NoData
            continue
        
        # Преобразуем индексы в географические координаты
        geo_x, geo_y = pixel_to_coords(j, i, gt)
        
        # Создаем точку и добавляем геометрии
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(geo_x, geo_y)

        # Создаем объект Feature
        feature = ogr.Feature(layer_defn)
        feature.SetGeometry(point)
        feature.SetField("ndvi", float(value))

        # Записываем точку в слой
        layer.CreateFeature(feature)

        # Освобождаем память
        feature = None
        point = None

# Закрытие shapefile
output_shapefile = None
