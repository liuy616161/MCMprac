from pyecharts import options as opts
from pyecharts.charts import Geo
from pyecharts.globals import ChartType,SymbolType
from pyecharts.charts import Map
from pyecharts.faker import Faker
from pyecharts.datasets import register_url
import os
register_url("https://echarts-maps.github.io/echarts-countries-js/")

c = (
    Geo()
    .add_schema(maptype="美国")
    .add_coordinate('CA', -120.44, 36.44)
    .add_coordinate('AZ',-111.70,34.45)
    .add_coordinate('NM',-106.11,34.45)
    .add_coordinate('WY',-107.29,42.96)
    .add_coordinate('CO',-105.52,38.89)
    .add_coordinate('LM',-114.39,36.25)
    .add_coordinate('BA',-111.29,36.56)
    .add("City",[list (z) for z in zip(["AZ",'CA','NM','CO','WY'],[3807.32,14791.83,1439.7,2697.83,2185.69])],type_=ChartType.EFFECT_SCATTER,)
    .add("Lake",[list(z) for z in zip(["LM","BA"],[13320000,11674000])])
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        visualmap_opts=opts.VisualMapOpts(max_=14000000,min_=11000000),
        title_opts=opts.TitleOpts(title="水库"),
    )
)
c.render()
os.system("render.html")
