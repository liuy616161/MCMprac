from pyecharts import options as opts
from pyecharts.charts import Map

# 江苏省各市的数据示例（可以根据实际情况替换成您的数据）
data = [("南京", 100), ("苏州", 150), ("无锡", 120), ("常州", 80), ("徐州", 90)]

# 创建地图
jiangsu_map = (
    Map()
    .add("江苏省", data, "江苏")
    .set_global_opts(
        title_opts=opts.TitleOpts(title="江苏省地图"),
        visualmap_opts=opts.VisualMapOpts(max_=200, is_piecewise=True)
    )
)

# 将地图渲染成 HTML 文件
jiangsu_map.render("jiangsu_map.html")
