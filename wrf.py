import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.feature as cfeature
from meteva.base.tool.plot_tools import add_china_map_2basemap
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
from tcmarkers import SH_TS  # 假设这是自定义的台风标记

# 读取文件
obs_tc = pd.read_excel('merged_data.xlsx')
obs_tc['纬度'] = obs_tc['纬度'] * 0.1
obs_tc['经度'] = obs_tc['经度'] * 0.1

# 添加地形
ter = xr.open_dataset(r'D:\code\gebco_2024_n62.0_s0.0_w95.0_e180.0.nc')
# 定义所需范围
lon_range = (95, 180)
lat_range = (0, 62)

# 提取指定范围的数据
ter_sub = ter.sel(lon=slice(*lon_range), lat=slice(*lat_range)).coarsen(lat=10, lon=10).mean()
h = ter_sub.elevation

# 绘图部分
extent = [95, 180, 0, 62]
fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(extent)

## 设置阴影
x, y = np.gradient(h.values)
slope = np.pi / 2. - np.arctan(np.sqrt(x * x + y * y))
# -x here because of pixel orders in the SRTM tile
aspect = np.arctan2(-x, y)
altitude = np.pi / 4.
azimuth = np.pi / 2.
shaded = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos((azimuth - np.pi / 2.) - aspect)

# 绘制地形阴影
plt.imshow(shaded, extent=extent, transform=ccrs.PlateCarree(), cmap=plt.cm.terrain, alpha=0.5, origin='lower', zorder=2)

# 绘制海洋颜色
ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='skyblue', alpha=0.8, zorder=3)

# 绘制观测台风路径
# 提取不同的台风轨迹
unique_cyclones = obs_tc['气旋编号'].unique()

colors = plt.cm.Set3(np.linspace(0, 1, len(unique_cyclones)))  # 使用更区分的颜色列表

for i, cyclone in enumerate(unique_cyclones):
    cyclone_data = obs_tc[obs_tc['气旋编号'] == cyclone]

    # 获取该台风的经纬度数据
    cyclone_lat = cyclone_data['纬度'].dropna().tolist()
    cyclone_lon = cyclone_data['经度'].dropna().tolist()

    # 绘制该台风的轨迹
    ax.scatter(cyclone_lon, cyclone_lat, color=colors[i], marker=SH_TS, s=1, transform=ccrs.PlateCarree(),
               label=f'Cyclone {cyclone}', zorder=4)
    ax.plot(cyclone_lon, cyclone_lat, color=colors[i], linewidth=0.5, linestyle='-', alpha=0.8, transform=ccrs.PlateCarree(), zorder=4)

#添加地理特征
add_china_map_2basemap(ax, name="river", edgecolor='k', lw=0.5, encoding='gbk', grid0=None)  # "河流"
add_china_map_2basemap(ax, name="nation", edgecolor='k', lw=0.5, encoding='gbk', grid0=None)  # "国界"
add_china_map_2basemap(ax, name="province", edgecolor='k', lw=0.5, encoding='gbk', grid0=None)  # "省界"

# 设置坐标轴刻度
ax.set_xticks(np.arange(extent[0], extent[1] + 1, 5), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(extent[-2], extent[-1] + 1, 5), crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
ax.yaxis.set_major_formatter(LatitudeFormatter())

# 设置横轴和纵轴的字体大小
ax.tick_params(axis='both', which='major', labelsize=5)

# 设置标题和图例的字体大小
plt.title('Typhoon Tracks', fontsize=8)

plt.show()