import geopandas as gpd
import contextily as ctx
import os
import rasterio as rio
from matplotlib import pyplot as plt
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
import fiona
import rasterio.mask
from rasterio.plot import show
from rasterio.enums import Resampling
import matplotlib.colors
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pandas.plotting import table
from unidecode import unidecode



class FixPointNormalize(matplotlib.colors.Normalize):
    """ 
    Inspired by https://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib
    Subclassing Normalize to obtain a colormap with a fixpoint 
    somewhere in the middle of the colormap.

    This may be useful for a `terrain` map, to set the "sea level" 
    to a color in the blue/turquise range. 
    """
    def __init__(self, vmin=None, vmax=None, sealevel=0, col_val = 0.21875, clip=False):
        # sealevel is the fix point of the colormap (in data units)
        self.sealevel = sealevel
        # col_val is the color value in the range [0,1] that should represent the sealevel.
        self.col_val = col_val
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.sealevel, self.vmax], [0, self.col_val, 1]
        return np.ma.masked_array(np.interp(value, x, y))


# Combine the lower and upper range of the terrain colormap with a gap in the middle
# to let the coastline appear more prominently.
# inspired by https://stackoverflow.com/questions/31051488/combining-two-matplotlib-colormaps
colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 56))
colors_land = plt.cm.terrain(np.linspace(0.25, 1, 200))
# combine them and build a new colormap
colors = np.vstack((colors_undersea, colors_land))
cut_terrain_map = matplotlib.colors.LinearSegmentedColormap.from_list('cut_terrain', colors)

#---------- Dictionary for contaminants
dic = {
       'Aluminio_d': ['Aluminio', '$mg/L$'],
       'Aluminio_t': ['Aluminio total', '$mg/L$'],
       'Arsenico_t': ['Arsénico total', '$mg/L$'],
       'Bicarbonat': ['Bicarbonato', '$mg/L$'],
       'Boro_total': ['Boro total', '$mg/L$'],
       'Cadmio_tot': ['Cadmio total', '$mg/L$'],
       'Calcio_tot': ['Calcio total', '$mg/L$'],
       'Carbonato': ['Carbonato', '$mg/L$'],
       'Cianuro_to': ['Cianuro total', '$mg/L$'],
       'Cloruro_to': ['Cloruro total', '$mg/L$'],
       'Cobalto_to': ['Cobalto total', '$mg/L$'],
       'Cobre_disu': ['Cobre disuelto', '$mg/L$'],
       'Cobre_tota': ['Cobre total', '$mg/L$'],
       'Coliformes': ['Coliformes', 'NMP/100mL'],
       'Coliform_1': ['Coliformes', 'NMP/100mL'],
       'Compuestos': ['Compuestos', '$mg/L$'],
       'Conductivi': ['Conductividad específica a 25$^{\circ}C$', '$\mu S/cm$'],
       'Cromo_hex_': ['Cromo hexavalente', '$mg/L$'],
       'Cromo_hex1': ['Cromo hexavalente', '$mg/L$'],
       'Cromo_tota': ['Cromo total', '$mg/L$'],
       'DBO5': ['Demanda Bioquímica de Oxígeno (DBO)', '$mg/L$'],
       'DQO': ['Demanda Química de Oxígeno', '$mg/L$'],
       'Dureza': ['Dureza', '$mg/L$'],
       'Fluoruro_t': ['Fluoruro total', '$mg/L$'],
       'Fosf_ortof': ['Fósforo de ortofosfato', '$mg/L$'],
       'Fosf_total': ['Fósforo total', '$mg/L$'],
       'Hierro_dis': ['Hierro disuelto', '$mg/L$'],
       'Hierro_tot': ['Hierro total', '$mg/L$'],
       'Litio_disu': ['Litio disuelto', '$mg/L$'],
       'Magnesio_t': ['Magnesio total', '$\mu g/L$'],
       'Manganeso_': ['Manganeso total', '$mg/L$'],
       'Manganeso1': ['Manganeso', '$mg/L$'],
       'Mercurio_t': ['Mercurio total', '$mg/L$'],
       'Molibdeno_': ['Molibdeno total', '$mg/L$'],
       'Niquel_dis': ['Níquel disuelto', '$mg/L$'],
       'Niquel_tot': ['Niquel total', '$mg/L$'],
       'Nitrogeno_': ['Nitrógeno', '$mg/L$'],
       'Nitrogeno1': ['Nitrógeno', '$mg/L$'],
       'Nitrogen_1': ['Nitrógeno', '$mg/L$'],
       'Nitrogen_2': ['Nitrógeno', '$mg/L$'],
       'Nitrogen_3': ['Nitrógeno', '$mg/L$'],
       'Nitrogen_4': ['Nitrógeno', '$mg/L$'],
       'Oxigeno_di': ['Oxígeno disuelto', '$mg/L$'],
       'pH': ['pH', 'Escala de pH'],
       'Plata_tota': ['Plata total', '$mg/L$'],
       'Plomo_disu': ['Plomo disuelto', '$mg/L$'],
       'Plomo_tota': ['Plomo total', '$mg/L$'],
       'Potasio_to': ['Potasio total', '$mg/L$'],
       'RAS': ['Relación de Absorción de Sodio (RAS)', '-'],
       'Selenio_to': ['Selenio total', '$mg/L$'],
       'Silice': ['Sílice', '$mg/L$'],
       'Sodio_tota': ['Sodio total', '$mg/L$'],
       'Solidos_su': ['Sólidos suspendidos', '$mg/L$'],
       'Sulfato': ['Sulfato', '$mg/L$'],
       'Temperatur': ['Temperatura del agua', '$^{\circ}C$'],
       'Zinc_disue': ['Zinc disuelto', '$mg/L$'],
       'Zinc_total': ['Zinc total', '$mg/L$']       
       }

def raster_resample(src,dst,upscale_factor):
    '''
    Resamples a source raster into dest raster by upscale_factor
    src: str, path to a raster file rasterio.open()
    dst: str, path to resampled raster file rasterio.open()
    upscale_factor: float, >1 for more detail (upscale)
    '''
    with rio.open(src) as dataset:
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor)
            ),
            resampling=Resampling.bilinear
        )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
        out_meta = dataset.meta
        out_meta.update({'driver': 'GTiff',
                            'height': data.shape[1],
                            'width': data.shape[2],
                            'transform': transform})
        with rio.open(dst, 'w', **out_meta) as dest:
            dest.write(data)

def raster_reproject(srcfile, dstfile, crs_dst):
    with rio.open(srcfile) as src:
        transform, width, height = calculate_default_transform(
            src.crs, crs_dst, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': crs_dst,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rio.open(dstfile, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=crs_dst,
                    resampling=Resampling.nearest)

def raster_mask(srcfile, dstfile, maskfile):
    with fiona.open(maskfile, 'r') as shapefile:
        shapes = [feature['geometry'] for feature in shapefile]
        
        with rio.open(srcfile) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True,
                                nodata=np.nan)
            out_meta = src.meta
            out_meta.update({'driver': 'GTiff',
                            'height': out_image.shape[1],
                            'width': out_image.shape[2],
                            'transform': out_transform,
                            'nodata': np.nan})
            with rio.open(dstfile, 'w', **out_meta) as dest:
                dest.write(out_image)


def raster_mask_Uint16(srcfile, dstfile, maskfile):
    with fiona.open(maskfile, 'r') as shapefile:
        shapes = [feature['geometry'] for feature in shapefile]
        
        with rio.open(srcfile) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True,
                                nodata=15999)
            out_meta = src.meta
            out_meta.update({'driver': 'GTiff',
                            'height': out_image.shape[1],
                            'width': out_image.shape[2],
                            'transform': out_transform,
                            'nodata': 15999})
            with rio.open(dstfile, 'w', **out_meta) as dest:
                dest.write(out_image)
                
def get_gdb_layers(gdbfolderpath):
    layers = fiona.listlayers(gdbfolderpath)
    for lay in layers:
        print(lay)
        
def import_gdb_layer(gdbfolderpath, layer): 
    gdf = gpd.read_file(gdbfolderpath, layer = layer)
    return gdf
            
def filter_gdf_by_gdf(gdf1,gdf2,crs):
    gdf1 = gdf1.to_crs(crs)
    gdf2 = gdf2.to_crs(crs)
    gdf1 = gpd.sjoin(gdf1,gdf2, how = 'inner')
    return gdf1

def raster_mask_by_geom(srcfile, dstfile, geometry):
    
    shapes = [geometry]
        
    with rio.open(srcfile) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True,
                            nodata=-999)
        # possible drivers: JP2OpenJPEG, GTiff
        out_meta = src.meta
        out_meta.update({'driver': 'GTiff',
                        'height': out_image.shape[1],
                        'width': out_image.shape[2],
                        #'dtype': 'float32',
                        'transform': out_transform,
                        'nodata': -999})
        #out_image = out_image.astype(np.float32)
        #out_image[out_image==0]=0
        
        with rio.open(dstfile, 'w', **out_meta) as dest:
            dest.write(out_image)

def raster_mask_by_geom_Uint16(srcfile, dstfile, geometry):
    
    shapes = [geometry]
        
    with rio.open(srcfile) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True,
                            nodata=15999)
        # possible drivers: JP2OpenJPEG, GTiff
        out_meta = src.meta
        out_meta.update({'driver': 'GTiff',
                        'height': out_image.shape[1],
                        'width': out_image.shape[2],
                        #'dtype': 'float32',
                        'transform': out_transform,
                        'nodata': 15999})
        #out_image = out_image.astype(np.float32)
        #out_image[out_image==0]=0
        
        with rio.open(dstfile, 'w', **out_meta) as dest:
            dest.write(out_image)


def plot_temperatures(rasterfp, options, name, ext_img):
    '''
    

    Parameters
    ----------
    rasterfp : STRING filepath of raster
        DESCRIPTION.
    options : [boolean, path (True) or geometry (False)]
        DESCRIPTION.
    name : STRING name of catchment to be plotted alongside raster
        DESCRIPTION.
    ext_img : STRING
        'jpg/pdf/etc'.

    Returns
    -------
    None.

    '''
    #tempfp = os.path.join(os.path.pardir(rasterfp), 'temp.tif')
    tempfp = 'temp.tif'
    usepath = options[0]
    path = options[1]
    
    if usepath:
        raster_mask(rasterfp, tempfp, path)
    else:
        raster_mask_by_geom(rasterfp, tempfp, options[1])
    
    fs = (8.5,11)
    fc = 'none'
    ec = 'black'
    ls = '--'
    ctlevels = 8
    provider = ctx.providers.Esri.WorldTerrain
    colors = 'black'
    cmap = 'coolwarm'
    zoom = 9
    with rio.open(tempfp) as src:
        fig = plt.figure(figsize = fs)
        ax = fig.add_subplot(111)
        if usepath:
            gdf = gpd.read_file(path)
            gdf.plot(ax = ax, fc=fc, ec = ec, ls=ls)
        else:
            p = gpd.GeoSeries(options[1])
            p.plot(ax = ax, fc=fc, ec = ec, ls=ls)
        ctx.add_basemap(ax = ax, crs = 'EPSG:32719',
                        source = provider, zoom=zoom)
        
        filt = src.read()>-273
        vmin = src.read()[filt].min()
        vmax = src.read()[filt].max()
        if vmin <0:
            norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        
        sm = ScalarMappable(norm = norm, cmap = cmap)
        show(src, ax = ax, contour=False, cmap=cmap, norm = norm)
        show(src, ax = ax, contour=True, colors=colors, levels=ctlevels,
             contour_label_kws={'fmt': '%1.0f',
                                'fontsize': 'medium'})
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.5)
        cbar = fig.colorbar(sm, cax=cax, shrink=1.0)
        cbar.ax.set_ylabel('Temperatura ($^{\circ}C$)')
        ax.set_title('\n'.join(['Temperatura media anual',
                                name]))
        ax.set_xlabel('Coordenada Este UTM (m)')
        ax.set_ylabel('Coordenada Norte UTM (m)')
        folder_saved_imgs_tt = os.path.join('./','outputs',
                                            'caracter_hidr', 'tt')
        filename = os.path.join(folder_saved_imgs_tt,
                                'tt_media_' + name +  '.' + ext_img)
        plt.savefig(filename, format=ext_img, bbox_inches = 'tight',
                    pad_inches = 0.1)
        plt.close()
        
        
def plot_precipitation(rasterfp, options, name, ext_img):
    '''
    

    Parameters
    ----------
    rasterfp : STRING filepath of raster
        DESCRIPTION.
    options : [boolean, path (True) or geometry (False)]
        DESCRIPTION.
    name : STRING name of catchment to be plotted alongside raster
        DESCRIPTION.
    ext_img : STRING
        'jpg/pdf/etc'.

    Returns
    -------
    None.

    '''
    #tempfp = os.path.join(os.path.pardir(rasterfp), 'temp.tif')
    tempfp = 'temp.tif'
    usepath = options[0]
    path = options[1]
    
    if usepath:
        raster_mask(rasterfp, tempfp, path)
    else:
        raster_mask_by_geom(rasterfp, tempfp, options[1])
    
    fs = (8.5,11)
    fc = 'none'
    ec = 'black'
    ls = '--'
    ctlevels = 6
    provider = ctx.providers.Esri.WorldTerrain
    colors = 'black'#'#edb51a'
    cmap = 'Blues'
    zoom = 9
    with rio.open(tempfp) as src:
        fig = plt.figure(figsize = fs)
        ax = fig.add_subplot(111)
        if usepath:
            gdf = gpd.read_file(path)
            gdf.plot(ax = ax, fc=fc, ec = ec, ls=ls)
        else:
            p = gpd.GeoSeries(options[1])
            p.plot(ax = ax, fc=fc, ec = ec, ls=ls)
        ctx.add_basemap(ax = ax, crs = 'EPSG:32719',
                        source = provider, zoom=zoom)
        
        filt = src.read()>-1
        # vmin = src.read()[filt].min()
        vmin = 0
        vmax = src.read()[filt].max()
        if vmin <0:
            norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        
        sm = ScalarMappable(norm = norm, cmap = cmap)
        show(src, ax = ax, contour=False, cmap=cmap, norm = norm)
        show(src, ax = ax, contour=True, colors=colors, levels=ctlevels,
             contour_label_kws={'fmt': '%1.0f',
                                'fontsize': 'medium'})
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.5)
        cbar = fig.colorbar(sm, cax=cax, shrink=1.0)
        cbar.ax.set_ylabel('Precipitacion (mm/año)')
        ax.set_title('\n'.join(['Precipitacion media anual',
                                name]))
        ax.set_xlabel('Coordenada Este UTM (m)')
        ax.set_ylabel('Coordenada Norte UTM (m)')
        folder_saved_imgs_pp = os.path.join('./','outputs',
                                            'caracter_hidr', 'pp')
        filename = os.path.join(folder_saved_imgs_pp,
                                'pp_media_' + name +  '.' + ext_img)
        plt.savefig(filename, format=ext_img, bbox_inches = 'tight',
                    pad_inches = 0.1)
        plt.close()

def plot_water_rights(daageodataframe, options, name, ext_img, value):
    dicvalue = {'fuente': 'Naturaleza_Agua_Clasificacion',
                'uso': 'Tipo_Derecho'}
    
    dictitle = {'fuente': 'Superficial vs subterránea',
                'uso': 'Consuntivo vs no-consuntivo'}
    
    dicheader = {'fuente': 'Tipo de fuente',
                 'uso': 'Tipo de uso'}
    
    fs = (11,8.5)
    fc = 'none'
    ec = 'black'
    ls = '--'
    size = 3
    provider = ctx.providers.Esri.WorldTerrain
    zoom = 9
    lw = 0.2
    cmap2 = 'Set3'
    loc2 = 4
    loc3 = 3
    
    if options[0]:
        daageodataframe = gpd.sjoin(daageodataframe, options[1], how='inner')
        hgfplot = gpd.sjoin(hgf, options[1], how = 'inner')
        lw = 0.2
    else:
        p = gpd.GeoSeries(options[1], crs = 'EPSG:32719')
        A = gpd.GeoDataFrame(p)
        A.rename({0: 'geometry'}, axis = 1, inplace=True)
        A.crs = 'EPSG:32719'
        daageodataframe = gpd.sjoin(daageodataframe,
                                    A, how = 'inner')
        hgfplot = gpd.sjoin(hgf,A, how = 'inner')
        size = 4
        lw=0.5

    fig = plt.figure(figsize = fs)
    ax1 = fig.add_subplot(121)
    if options[0]:
        options[1].plot(ax = ax1, fc = fc, ec = ec, ls = ls)
    else:
        p.plot(ax = ax1, fc = fc, ec = ec, ls = ls)
    ctx.add_basemap(ax = ax1, crs = 'EPSG:32719',
                        source = provider, zoom=zoom)
    hgfplot.geometry.plot(ax = ax1, color = 'blue',
                          linewidth = lw, linestyle = '--', alpha=0.5)
    daageodataframe.plot(ax = ax1, markersize= size)
    # hgfplot.geometry.plot(ax = ax1, color = 'teal',
    #                       linewidth = lw, linestyle = '--', alpha=0.4)
    ax1.set_title('\n'.join(['DAA Totales']))
    ax1.set_xlabel('Coordenada Este UTM (m)')
    ax1.set_ylabel('Coordenada Norte UTM (m)')
    
    # Ax 2: the one plotting classes
    
    ax2 = fig.add_subplot(122)
    nanfilter = daageodataframe[dicvalue[value]] != 'nan'
    daageodataframe[nanfilter].plot(ax = ax2, markersize= size,
                         column = dicvalue[value],
                         legend=True, cmap = cmap2)
    if options[0]:
        options[1].plot(ax = ax2, fc = fc, ec = ec, ls = ls)
    else:
        p.plot(ax = ax2, fc = fc, ec = ec, ls = ls)
    ctx.add_basemap(ax = ax2, crs = 'EPSG:32719',
                        source = provider, zoom=zoom)
    hgfplot.geometry.plot(ax = ax2, color = 'blue',
                          linewidth = lw, linestyle = '--', alpha=0.5)
    
    # inset ax2
    inax2 = inset_axes(ax2,
                    width="20%", # width = 30% of parent_bbox
                    height='20%', # height : 1 inch
                    loc=loc2)
    # inax3 = inset_axes(ax2,
    #                 width="40%", # width = 30% of parent_bbox
    #                 height='10%', # height : 1 inch
    #                 loc=loc3)
    vcounts = daageodataframe[daageodataframe[dicvalue[value]]!= 'nan'][dicvalue[value]]
    vcountstable = vcounts.value_counts()
    vcounts = vcounts.value_counts(normalize=True)
    table(ax = ax2, data=vcountstable, loc='bottom', colWidths=[0.6,0.4],
          colLabels=[dicheader[value]])
    vcounts.plot(ax=inax2, kind='pie', use_index=False, colormap = cmap2,
                 autopct='%1.0f%%', labels = None, fontsize = 8)
    plt.subplots_adjust(bottom=0.2)
    inax2.set_ylabel('')
    
    ax2.set_title('DAA ' + dictitle[value])
    #ax2.set_xlabel('Coordenada Este UTM (m)')
    ax2.set_xlabel('')
    ax2.set_xticks([])
    ax2.set_ylabel('Coordenada Norte UTM (m)')
    
    
    folder_saved_imgs_pp = os.path.join('./','outputs',
                                            'caracter_hidr', 'DAA')
    
    plt.suptitle('\n'.join(['Derechos de Aprovechamiento de Aguas',
                            name]))
    name = unidecode(name)
    name = name.replace(' ','')
    filename = os.path.join(folder_saved_imgs_pp,
                                'DAA_' + value + '_' + name +  '.' + ext_img)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, format=ext_img, bbox_inches = 'tight',
                pad_inches = 0.1)
    
    # plt.savefig(filename, format=ext_img)
    plt.close()
    
def plot_water_rights2(daageodataframe, options, name, ext_img, value):
    # dicvalue = {'fuente': 'Naturaleza_Agua_Clasificacion',
    #             'uso': 'Tipo_Derecho'}
    dicvalue = {'fuente': 'Naturale_1',
                'uso': 'Tipo_Derec'}
    
    dictitle = {'fuente': 'Superficial vs subterránea',
                'uso': 'Consuntivo vs no-consuntivo'}
    
    dicheader = {'fuente': 'Tipo de fuente',
                 'uso': 'Tipo de uso'}
    
    diccolors = {'fuente': {'Superficial':'green', 'Subterranea':'red'},
                 'uso': {'Consuntivo':'red', 'No Consuntivo':'green'}}
    dic_g = {'fuente': 'Superficial', 'uso': 'No Consuntivo'}
    dic_r = {'fuente': 'Subterranea', 'uso': 'Consuntivo'}
    
    fs = (11,8.5)
    fc = 'none'
    ec = 'black'
    ls = '--'
    size = 1
    provider = ctx.providers.Esri.WorldTerrain
    zoom = 9
    lw = 0.2
    cmap2 = 'Set3'
    loc2 = 4
    loc3 = 3
    
    if options[0]:
        daageodataframe = gpd.sjoin(daageodataframe, options[1], how='inner')
        hgfplot = gpd.sjoin(hgf, options[1], how = 'inner')
        lw = 0.2
    else:
        p = gpd.GeoSeries(options[1], crs = 'EPSG:32719')
        A = gpd.GeoDataFrame(p)
        A.rename({0: 'geometry'}, axis = 1, inplace=True)
        A.crs = 'EPSG:32719'
        daageodataframe = gpd.sjoin(daageodataframe,
                                    A, how = 'inner')
        hgfplot = gpd.sjoin(hgf,A, how = 'inner')
        size = 4
        lw=0.5

    fig = plt.figure(figsize = fs)
    # ax1 = fig.add_subplot(121)
    # if options[0]:
    #     options[1].plot(ax = ax1, fc = fc, ec = ec, ls = ls)
    # else:
    #     p.plot(ax = ax1, fc = fc, ec = ec, ls = ls)
    # ctx.add_basemap(ax = ax1, crs = 'EPSG:32719',
    #                     source = provider, zoom=zoom)
    # daageodataframe.plot(ax = ax1, markersize= size)
    # # hgfplot.geometry.plot(ax = ax1, color = 'teal',
    # #                       linewidth = lw, linestyle = '--', alpha=0.4)
    # ax1.set_title('\n'.join(['DAA Totales']))
    # ax1.set_xlabel('Coordenada Este UTM (m)')
    # ax1.set_ylabel('Coordenada Norte UTM (m)')
    
    # Ax 2: the one plotting classes
    
    ax2 = fig.add_subplot(111)
    # nanfilter = daageodataframe[dicvalue[value]] != 'nan'
    # daageodataframe[nanfilter].plot(ax = ax2, markersize= size,
    #                      column = dicvalue[value],
    #                      legend=True, cmap = cmap2)
    nanfilter_g = (daageodataframe[dicvalue[value]] != 'nan') & \
        daageodataframe[dicvalue[value]].isin(['Superficial', 'No Consuntivo'])
    daageodataframe[nanfilter_g].plot(ax = ax2, markersize= size,
                         column = dicvalue[value],
                         label=dic_g[value], color = 'green')
    nanfilter_r = (daageodataframe[dicvalue[value]] != 'nan') & \
        daageodataframe[dicvalue[value]].isin(['Subterranea', 'Consuntivo'])
    daageodataframe[nanfilter_r].plot(ax = ax2, markersize= size,
                         column = dicvalue[value],
                         label=dic_r[value], color = 'red')
    
    
    
    if options[0]:
        options[1].plot(ax = ax2, fc = fc, ec = ec, ls = ls)
    else:
        p.plot(ax = ax2, fc = fc, ec = ec, ls = ls)
    ctx.add_basemap(ax = ax2, crs = 'EPSG:32719',
                        source = provider, zoom=zoom)
    hgfplot.geometry.plot(ax = ax2, color = 'blue',
                          linewidth = lw, linestyle = '--')
    
    # inset ax2
    inax2 = inset_axes(ax2,
                    width="20%", # width = 30% of parent_bbox
                    height='20%', # height : 1 inch
                    loc=loc2)
    # inax3 = inset_axes(ax2,
    #                 width="40%", # width = 30% of parent_bbox
    #                 height='10%', # height : 1 inch
    #                 loc=loc3)
    vcounts = daageodataframe[daageodataframe[dicvalue[value]]!= 'nan'][dicvalue[value]]
    vcountstable = vcounts.value_counts()
    vcounts = vcounts.value_counts(normalize=True)
    table(ax = ax2, data=vcountstable, loc='bottom', colWidths=[0.6,0.4],
          colLabels=[dicheader[value]])
    vcounts.plot(ax=inax2, kind='pie', use_index=False,
                 autopct='%1.0f%%', labels = None, fontsize = 8,
                 colors=[diccolors[value][v] for v in vcounts.keys()])
    plt.subplots_adjust(bottom=0.2)
    inax2.set_ylabel('')
    ax2.legend()
    ax2.set_title('DAA ' + dictitle[value])
    #ax2.set_xlabel('Coordenada Este UTM (m)')
    ax2.set_xlabel('')
    ax2.set_xticks([])
    ax2.set_ylabel('Coordenada Norte UTM (m)')
    
    
    folder_saved_imgs_pp = os.path.join('./','outputs',
                                            'caracter_hidr', 'DAA')
    
    plt.suptitle('\n'.join(['Derechos de Aprovechamiento de Aguas',
                            name]))
    name = unidecode(name)
    name = name.replace(' ', '')
    filename = os.path.join(folder_saved_imgs_pp,
                                'DAA_' + value + '_' + name +  '.' + ext_img)
    plt.savefig(filename, format=ext_img, bbox_inches = 'tight',
                pad_inches = 0.1)
    # plt.savefig(filename, format=ext_img)
    plt.close()
    
    
def plot_DEM(rasterfp, options, name, ext_img):
    '''
    

    Parameters
    ----------
    rasterfp : STRING filepath of raster
        DESCRIPTION.
    options : [boolean, path (True) or geometry (False)]
        DESCRIPTION.
    name : STRING name of catchment to be plotted alongside raster
        DESCRIPTION.
    ext_img : STRING
        'jpg/pdf/etc'.

    Returns
    -------
    None.

    '''
    #tempfp = os.path.join(os.path.pardir(rasterfp), 'temp.tif')
    tempfp = 'dem.tif'
    usepath = options[0]
    path = options[1]
    
    if usepath:
        raster_mask_Uint16(rasterfp, tempfp, path)
    else:
        raster_mask_by_geom_Uint16(rasterfp, tempfp, options[1])
    
    fs = (8.5,11)
    fc = 'none'
    ec = 'black'
    ls = '--'
    ctlevels = 6
    provider = ctx.providers.Esri.WorldTerrain
    colors = 'orange'#'#edb51a'
    cmap = 'terrain'
    zoom = 9
    with rio.open(tempfp) as src:
        fig = plt.figure(figsize = fs)
        ax = fig.add_subplot(111)
        if usepath:
            gdf = gpd.read_file(path)
            gdf.plot(ax = ax, fc=fc, ec = ec, ls=ls)
        else:
            p = gpd.GeoSeries(options[1])
            p.plot(ax = ax, fc=fc, ec = ec, ls=ls)
        ctx.add_basemap(ax = ax, crs = 'EPSG:32719',
                        source = provider, zoom=zoom)
        
        filt = (src.read()<10000) & (src.read()>0)
        vmin = src.read()[filt].min()
        print(vmin)
        
        vmax = src.read()[filt].max()
        print(vmax)
        if vmin <0:
            norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
        else:
            # norm = Normalize(vmin=vmin, vmax=vmax)
            norm = LogNorm(vmin=vmin, vmax=10000)
            # norm = FixPointNormalize(sealevel=0, vmax=vmax)
            
        
        sm = ScalarMappable(norm = norm, cmap = cmap)
        show(src, ax = ax, contour=False, cmap=cmap, norm = norm)
        # show(src, ax = ax, contour=True, colors=colors, levels=ctlevels,
        #      contour_label_kws={'fmt': '%1.0f',
        #                         'fontsize': 'medium'})
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.5)
        cbar = fig.colorbar(sm, cax=cax, shrink=1.0)
        cbar.ax.set_ylabel('Elevación (msnm)')
        ax.set_title('\n'.join(['Elevacion_Terreno',
                                name]))
        ax.set_xlabel('Coordenada Este UTM (m)')
        ax.set_ylabel('Coordenada Norte UTM (m)')
        folder_saved_imgs_pp = os.path.join('./','outputs',
                                            'caracter_hidr', 'dem')
        filename = os.path.join(folder_saved_imgs_pp,
                                'dem_' + name +  '.' + ext_img)
        plt.savefig(filename, format=ext_img, bbox_inches = 'tight',
                    pad_inches = 0.1)
        plt.close()
        
        
def plot_cr2Temp(rasterfp, options, name, ext_img):
    '''
    

    Parameters
    ----------
    rasterfp : STRING filepath of raster
        DESCRIPTION.
    options : [boolean, path (True) or geometry (False)]
        DESCRIPTION.
    name : STRING name of catchment to be plotted alongside raster
        DESCRIPTION.
    ext_img : STRING
        'jpg/pdf/etc'.

    Returns
    -------
    None.

    '''
    #tempfp = os.path.join(os.path.pardir(rasterfp), 'temp.tif')
    tempfp = 'cr2temp.tif'
    usepath = options[0]
    path = options[1]
    
    if usepath:
        raster_mask_Uint16(rasterfp, tempfp, path)
    else:
        raster_mask_by_geom_Uint16(rasterfp, tempfp, options[1])
    
    fs = (8.5,11)
    fc = 'none'
    ec = 'black'
    ls = '--'
    ctlevels = 6
    provider = ctx.providers.Esri.WorldTerrain
    colors = 'orange'#'#edb51a'
    cmap = 'coolwarm'
    zoom = 9
    band = 5000
    with rio.open(tempfp) as src:
        fig = plt.figure(figsize = fs)
        ax = fig.add_subplot(111)
        if usepath:
            gdf = gpd.read_file(path)
            gdf.plot(ax = ax, fc=fc, ec = ec, ls=ls)
        else:
            p = gpd.GeoSeries(options[1])
            p.plot(ax = ax, fc=fc, ec = ec, ls=ls)
        ctx.add_basemap(ax = ax, crs = 'EPSG:32719',
                        source = provider, zoom=zoom)
        
        filt = (src.read(band)<150) & (src.read(band)>-50)
        vmin = src.read(band)[filt].min()
        print(vmin)
        
        vmax = src.read(band)[filt].max()
        print(vmax)
        if vmin <0:
            norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
        else:
            # norm = Normalize(vmin=vmin, vmax=vmax)
            norm = LogNorm(vmin=vmin, vmax=vmax)
            # norm = FixPointNormalize(sealevel=0, vmax=vmax)
            
        
        sm = ScalarMappable(norm = norm, cmap = cmap)
        show(src, ax = ax, contour=False, cmap=cmap, norm = norm)
        # show(src, ax = ax, contour=True, colors=colors, levels=ctlevels,
        #      contour_label_kws={'fmt': '%1.0f',
        #                         'fontsize': 'medium'})
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.5)
        cbar = fig.colorbar(sm, cax=cax, shrink=1.0)
        cbar.ax.set_ylabel('Elevación (msnm)')
        ax.set_title('\n'.join(['Temperatura cr2MET',
                                name]))
        ax.set_xlabel('Coordenada Este UTM (m)')
        ax.set_ylabel('Coordenada Norte UTM (m)')
        folder_saved_imgs_pp = os.path.join('./','outputs',
                                            'caracter_hidr', 'cr2MET_temp')
        filename = os.path.join(folder_saved_imgs_pp,
                                'cr2met_temp_' + name +  '.' + ext_img)
        plt.savefig(filename, format=ext_img, bbox_inches = 'tight',
                    pad_inches = 0.1)
        plt.close()
#%%
#-------------- Define folders and filepaths
folder_BH = os.path.join('..', 'SIG', 'REH5796_Proyecto_SIG_BH1', '4_Anexos',
                      '3_Archivos_raster', '3_Forzantes_1985-2015')
folder_catch = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Cuencas_DARH',
                          'Cuencas')
folder_saved_imgs_pp = os.path.join('./','outputs', 'caracter_hidr', 'pp')
folder_saved_imgs_tt = os.path.join('./','outputs', 'caracter_hidr', 'tt')
folder_saved_imgs_wq = os.path.join('./','outputs', 'caracter_hidr', 'wq')
folder_camels = os.path.join('..', 'SIG', 'Cuencas_CAMELS')
folder_CQ = os.path.join('..', 'SIG', 'CQA5868', 'CQA5868_GDB',
                         'Estaciones_DGA.gdb')
folder_hgf = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Hidrografia')
hgf_fp = os.path.join(folder_hgf, 'RedHidrograficaUTM19S_AOHIA_ZC.geojson')
hgf = gpd.read_file(hgf_fp)
# hgf.drop('index_right', axis=1, inplace=True)


# paths for temperature
tmafile = os.path.join(folder_BH,'Tma_85-15_Chile.tif')
tma_reproj_file = os.path.join(folder_BH,'Tma_85-15_Chile_epsg32719.tif')
tma_rsed_file = os.path.join(folder_BH,'Tma_85-15_Chile_epsg32719_AOHIA_ZC_rs.tif')
tma_masked_file = os.path.join(folder_BH,'Tma_85-15_Chile_epsg32719_AOHIA_ZC.tif')


# paths for precipitation
pmafile = os.path.join(folder_BH, 'Pma_85-15_Chile.tif')
pma_reproj_file = os.path.join(folder_BH,'Pma_85-15_Chile_epsg32719.tif')
pma_rsed_file = os.path.join(folder_BH,'Pma_85-15_Chile_epsg32719_AOHIA_ZC_rs.tif')
pma_masked_file = os.path.join(folder_BH,'Pma_85-15_Chile_epsg32719_AOHIA_ZC.tif')

# path for catchments
catchments = os.path.join(folder_catch, 'Cuencas_DARH_2015_AOHIA_ZC.geojson')
gdf_catchments = gpd.read_file(catchments)
catchments_camels = os.path.join(folder_camels,
                                 'Cuencas_cabecera_MaipoRapelMataquitoMaule.shp')
gdf_camels = gpd.read_file(catchments_camels)
gdf_camels = gdf_camels.to_crs('epsg:32719')
catchments_camels = os.path.join(folder_camels,
                                 'Cuencas_cabecera_MaipoRapelMataquitoMaule_epsg32719.shp')
gdf_camels.to_file(catchments_camels)

# path for Water Rights geodataframe
# daafolder = os.path.join('..', 'Etapa 1 y 2', 'DAA', 'shapes_output')
daafolder = os.path.join('..', 'Etapa 1 y 2', 'DAA', 'shapes_output',
                         'Act_2021_05_15')
# daafp = os.path.join(daafolder, 'captaciones_post_ALL_CFA.geojson')
daafp = os.path.join(daafolder, 'DAA.shp')
daa = gpd.read_file(daafp)

# path for DEM
demfolder = os.path.join('..', 'Etapa 1 y 2', 'DEM')
demfp = os.path.join(demfolder, 'DEM Alos 5a a 8a mar.jp2')
dem_masked_dst_folder = os.path.join('./','outputs', 'caracter_hidr', 'dem')
dem_rsed_file = os.path.join(dem_masked_dst_folder, 'DEM_APalsar_AOHIA_ZC_rsed.tif')
dem_masked_file = os.path.join(dem_masked_dst_folder, 'DEM_APalsar_AOHIA_ZC.tif')
dem_camels_masked_file = os.path.join(dem_masked_dst_folder, 'DEM_APalsar_camels.tif')

# path for cr2met temp
cr2folder = os.path.join('/home', 'faarrosp', 'Downloads')
cr2fp = os.path.join(cr2folder, 'cr2MET_temp.tif')
cr2_masked_file = os.path.join(cr2folder, 'cr2MET_temp_masked.tif')




#------------ Processes temperature
#raster_reproject(tmafile,tma_reproj_file,'EPSG:32719')
#raster_resample(tma_reproj_file,tma_rsed_file,10)
#raster_mask(tma_rsed_file,tma_masked_file,catchments)

# #------------ Processes precipitation
# raster_reproject(pmafile,pma_reproj_file,'EPSG:32719')
# raster_resample(pma_reproj_file,pma_rsed_file,10)
# raster_mask(pma_rsed_file,pma_masked_file,catchments)

# #------------ Processes DEM
# raster_reproject(pmafile,pma_reproj_file,'EPSG:32719')
# raster_resample(demfp,dem_rsed_file,0.1)
# raster_mask_Uint16(dem_rsed_file,dem_masked_file,catchments)
# raster_mask_Uint16(dem_rsed_file,dem_camels_masked_file,catchments_camels)

# raster_mask_Uint16(cr2fp,cr2_masked_file,catchments)

#%% -------------- Plotear Derechos de agua por cuenca

# -------- Macrocuencas
plot_water_rights2(daa, [True, gdf_catchments], 'Macrocuencas', 'pdf', 'fuente')
plot_water_rights2(daa, [True, gdf_catchments], 'Macrocuencas', 'pdf', 'uso')

# -------- Macrocuencas (individual)
for idx in gdf_catchments.index:
    geometry = gdf_catchments.loc[idx,'geometry']
    name = gdf_catchments.loc[idx, 'NOM_CUENCA']
    plot_water_rights2(daa, [False, geometry], name, 'pdf', 'fuente')
    plot_water_rights2(daa, [False, geometry], name, 'pdf', 'uso')

# -------- Cuencas cabecera
for idx in gdf_camels.index:
    geometry = gdf_camels.loc[idx,'geometry']
    name = 'cabec_' + gdf_camels.loc[idx, 'gauge_name']
    plot_water_rights2(daa, [False, geometry], name, 'pdf', 'fuente')
    plot_water_rights2(daa, [False, geometry], name, 'pdf', 'uso')

folder_daa = os.path.join('.','outputs', 'caracter_hidr', 'DAA')


filelist = [x for x in os.listdir(folder_daa) if ('DAA' in x and x.endswith('.pdf'))]
filelist.sort()
fp = os.path.join(folder_daa, 'Anexo_DAA.tex')

with open(fp, 'w') as f:
    for file in filelist:
        text = '\\includepdf[pages=-]{' + file +'}'
        f.write(text + '\n')


#%% -------------- Plotear rasters de temperatura

## -------- Macrocuencas
# plot_temperatures(tma_masked_file, [True, catchments], 'Macrocuencas',
#                   'jpg')

# # -------- Macrocuencas (individual)
# for idx in gdf_catchments.index:
#     geometry = gdf_catchments.loc[idx,'geometry']
#     name = gdf_catchments.loc[idx, 'NOM_CUENCA']
#     plot_temperatures(tma_masked_file, [False, geometry], name, 'jpg')
    

# # -------- Cuencas cabecera
# for idx in gdf_camels.index:
#     geometry = gdf_camels.loc[idx,'geometry']
#     name = 'cabec_' + gdf_camels.loc[idx, 'gauge_name']
#     plot_temperatures(tma_masked_file, [False, geometry], name, 'jpg')

#%% ------------- Plotear rasters de precipitacion

# # -------- Macrocuencas
# plot_precipitation(pma_masked_file, [True, catchments], 'Macrocuencas',
#                   'jpg')

# # -------- Macrocuencas (individual)
# for idx in gdf_catchments.index:
#     geometry = gdf_catchments.loc[idx,'geometry']
#     name = gdf_catchments.loc[idx, 'NOM_CUENCA']
#     plot_precipitation(pma_masked_file, [False, geometry], name, 'jpg')

# # -------- Cuencas cabecera
# for idx in gdf_camels.index:
#     geometry = gdf_camels.loc[idx,'geometry']
#     name = 'cabec_' + gdf_camels.loc[idx, 'gauge_name']
#     plot_precipitation(pma_masked_file, [False, geometry], name, 'jpg')


#%%

# #------------ Calidad del Agua geodatabase
# #get_gdb_layers(folder_CQ)
# gdf = import_gdb_layer(folder_CQ, 'Promedios')
# gdf = filter_gdf_by_gdf(gdf, gdf_catchments, 'EPSG:32719')

# fs=(8.5,11)
# fc='none'
# ls='--'
# ec='black'
# zoom=9
# provider = ctx.providers.Esri.WorldTerrain
# crs='EPSG:32719'
# ext_img='jpg'


# for medicion in ['superficial', 'subterranea']:
#     subset0 = gdf[gdf['Medicion'].isin([medicion])].copy()
#     for col in subset0.columns[15:70]:
#         subset=subset0[subset0[col]>-999].copy()
#         try:
#             vmin, vmax = subset[col].min(), subset[col].max()
#             norm = LogNorm(vmin, vmax)
#             sm = ScalarMappable(norm = norm, cmap = 'magma_r')
#             fig = plt.figure(figsize=fs)
#             ax = fig.add_subplot(111)
#             ax.scatter(x=subset.geometry.x,
#                         y=subset.geometry.y,
#                         s=subset[col]/(vmax)*100,
#                         c=subset[col], norm=norm,cmap='magma_r')
#             divider = make_axes_locatable(ax)
#             cax = divider.append_axes("right", size="5%", pad=0.5)
#             cbar = fig.colorbar(sm, cax=cax, shrink=1.0)
#             cbar.ax.set_ylabel(dic[col][0] + ' ' + dic[col][1])
#             gdf_catchments.plot(ax=ax, fc=fc, ls=ls, ec=ec)
#             ctx.add_basemap(ax=ax,crs=crs,zoom=zoom, source=provider)
#             ax.set_title('Valores Promedio de ' + dic[col][0])
#             ax.set_xlabel('Coordenada Este UTM (m)')
#             ax.set_ylabel('Coordenada Norte UTM (m)')
#             filename = os.path.join(folder_saved_imgs_wq, medicion,
#                                     'wq_media_' + col  + '.' + ext_img)
            
#             plt.savefig(filename, format=ext_img, bbox_inches = 'tight',
#                         pad_inches = 0.1)
#             plt.close()
#         except:
#             plt.close()


#%% ------------- Plotear rasters de DEM

# # -------- Macrocuencas
# plot_DEM(dem_masked_file, [True, catchments], 'Macrocuencas',
#                   'jpg')

# # # -------- Macrocuencas (individual)
# for idx in gdf_catchments.index:
#     geometry = gdf_catchments.loc[idx,'geometry']
#     name = gdf_catchments.loc[idx, 'NOM_CUENCA']
#     plot_DEM(dem_masked_file, [False, geometry], name, 'jpg')

# # # -------- Cuencas cabecera
# for idx in gdf_camels.index:
#     geometry = gdf_camels.loc[idx,'geometry']
#     name = 'cabec_' + gdf_camels.loc[idx, 'gauge_name']
#     plot_DEM(dem_masked_file, [False, geometry], name, 'jpg')


    
#%% ------------- Plotear rasters de cr2MET temp

# -------- Macrocuencas
# plot_cr2Temp(cr2_masked_file, [True, catchments], 'Macrocuencas',
#                   'jpg')

# # # -------- Macrocuencas (individual)
# for idx in gdf_catchments.index:
#     geometry = gdf_catchments.loc[idx,'geometry']
#     name = gdf_catchments.loc[idx, 'NOM_CUENCA']
#     plot_DEM(dem_masked_file, [False, geometry], name, 'jpg')

# # # -------- Cuencas cabecera
# for idx in gdf_camels.index:
#     geometry = gdf_camels.loc[idx,'geometry']
#     name = 'cabec_' + gdf_camels.loc[idx, 'gauge_name']
#     plot_DEM(dem_masked_file, [False, geometry], name, 'jpg')
