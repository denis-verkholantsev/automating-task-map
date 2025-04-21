import geopandas as gpd
from shapely.ops import polygonize
from pathlib import Path


GEOM_TYPES_TO_RETURN = {"Polygon", 'MultiPolygon'}


def make_polygon_filepath(path: str) -> str:
    p = Path(path)
    return str(p.with_name(p.stem + '_polygon.shp'))


def lines_to_polygon(input_lines_shp: str, encoding: str = 'utf-8') -> str:
    gdf = gpd.read_file(input_lines_shp, encoding=encoding)

    all_lines = []
    for geom in gdf.geometry:
        if geom.geom_type in GEOM_TYPES_TO_RETURN:
            return input_lines_shp
        elif geom.geom_type == "LineString":
            all_lines.append(geom)
        elif geom.geom_type == "MultiLineString":
            all_lines.extend(list(geom.geoms))
        
    polygons = list(polygonize(all_lines))

    poly_gdf = gpd.GeoDataFrame(geometry=polygons, crs=gdf.crs)
    output_polygon_shp = make_polygon_filepath(input_lines_shp)
    poly_gdf.to_file(output_polygon_shp)

    return output_polygon_shp