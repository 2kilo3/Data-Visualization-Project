from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_io import read_csv_preserve_codes  # noqa: E402
from src.visuals import (  # noqa: E402
    TYPE_LABELS,
    build_country_edges_from_rows,
    build_country_detail_table,
    collaboration_scale_distribution,
    concept_cooccurrence_heatmap,
    concept_type_heatmap,
    concept_chart,
    filter_elements,
    filtered_country_rows,
    multinational_share_by_type,
    network_centrality_chart,
    network_figure,
    network_pair_chart,
    network_summary,
    rank_shift_chart,
    recompute_country_summary,
    region_type_heatmap,
    region_urgent_share_chart,
    safeguarding_pressure_focus_chart,
    top_countries,
    type_share_timeline,
    world_map,
    yearly_trend,
)


PROCESSED_DIR = ROOT / "data" / "processed"


st.set_page_config(
    page_title="UNESCO 非遗可视化图谱",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
body, .stApp {
    background: #F8FAFC;
    color: #1F2933;
}
.block-container {
    padding-top: 2rem;
    max-width: 1320px;
}
.block-container h1 {
    font-size: clamp(1.9rem, 3vw, 2.55rem);
    line-height: 1.28;
    letter-spacing: 0;
    margin: 0.2rem 0 0.9rem;
    overflow: visible;
}
.block-container h2, .block-container h3 {
    letter-spacing: 0;
}
[data-testid="stMetricValue"] {
    color: #1E3A8A;
    font-variant-numeric: tabular-nums;
}
.quiet-note {
    color: #52606D;
    font-size: 0.92rem;
    line-height: 1.5;
}
.section-lead {
    border-left: 4px solid #1E40AF;
    padding-left: 0.8rem;
    color: #334E68;
}
div[data-testid="stTabs"] button {
    font-size: 0.95rem;
}
div[data-baseweb="tag"] {
    background-color: #E9EEF6;
    color: #1E3A8A;
}
div[data-baseweb="tag"] span {
    color: #1E3A8A;
}
</style>
"""


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    return {
        "elements": read_csv_preserve_codes(PROCESSED_DIR / "elements.csv"),
        "element_countries": read_csv_preserve_codes(PROCESSED_DIR / "element_countries.csv"),
        "concepts": read_csv_preserve_codes(PROCESSED_DIR / "element_concepts.csv"),
        "countries": read_csv_preserve_codes(PROCESSED_DIR / "country_summary.csv"),
    }


def type_label(acronym: str) -> str:
    return TYPE_LABELS.get(acronym, acronym)


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    data = load_data()
    elements = data["elements"]
    element_countries = data["element_countries"]
    concepts = data["concepts"]
    countries = data["countries"]

    min_year = int(elements["inscription_year"].min())
    max_year = int(elements["inscription_year"].max())

    with st.sidebar:
        st.header("筛选器")
        year_range = st.slider(
            "入选年份范围",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
        )
        type_options = ["RL", "USL", "Art18"]
        selected_types = st.multiselect(
            "名录类型",
            options=type_options,
            default=type_options,
            format_func=type_label,
        )
        regions = sorted(countries["region"].dropna().unique())
        selected_regions = st.multiselect("World Bank 区域", options=regions, default=regions)
        metric_choice = st.radio(
            "地图与排名指标",
            options=["element_count", "urgent_share", "elements_per_million"],
            format_func=lambda value: {
                "element_count": "国家项目数量",
                "urgent_share": "急需保护占比",
                "elements_per_million": "每百万人项目数（审计）",
            }[value],
        )
        min_edge_weight = st.slider("网络最少共享项目数", 1, 8, 2)
        focus_country = st.selectbox(
            "国家详情",
            options=["全部国家"] + countries.sort_values("country_name")["country_name"].dropna().tolist(),
        )

    st.title("全球非物质文化遗产可见度与保护压力图谱")
    st.markdown(
        "<p class='section-lead'>本应用将 UNESCO 非遗名录记录与国家背景指标连接起来，"
        "用于观察不同国家的登记可见度、急需保护压力与跨国联合申报关系。地图默认展示国家项目数量，"
        "并保留每百万人项目数作为补充审计指标。</p>",
        unsafe_allow_html=True,
    )

    filtered_elements = filter_elements(elements, year_range, selected_types)
    filtered_country = filtered_country_rows(element_countries, filtered_elements)
    active_country_summary = recompute_country_summary(filtered_country, countries)
    if selected_regions:
        active_country_summary = active_country_summary[active_country_summary["region"].isin(selected_regions)]
        filtered_country = filtered_country[filtered_country["iso2"].isin(active_country_summary["iso2"])]
        filtered_elements = filtered_elements[filtered_elements["element_id"].isin(filtered_country["element_id"])]

    active_edges = build_country_edges_from_rows(filtered_country)
    visible_edges = active_edges[active_edges["weight"] >= min_edge_weight]
    network_stats = network_summary(visible_edges)
    total_elements = filtered_elements["element_id"].nunique()
    total_countries = filtered_country["iso2"].nunique()
    urgent_count = int((filtered_elements["type_acronym"] == "USL").sum())
    multi_count = int(active_edges["shared_elements"].str.split(";").apply(len).sum()) if not active_edges.empty else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("非遗项目", f"{total_elements:,}")
    k2.metric("国家 / 地区", f"{total_countries:,}")
    k3.metric("急需保护项目", f"{urgent_count:,}")
    k4.metric("跨国关系边", f"{len(active_edges):,}", help=f"当前筛选下共涉及 {multi_count:,} 条共享申报记录")

    if filtered_elements.empty or filtered_country.empty:
        st.warning("当前筛选条件下没有记录。请扩大年份范围或增加名录类型。")
        return

    tab_map, tab_time, tab_theme, tab_network, tab_country = st.tabs(
        ["全球地图", "时间演化", "主题结构", "跨国网络", "国家详情"]
    )

    with tab_map:
        st.subheader("地图与排名")
        st.plotly_chart(world_map(active_country_summary, metric_choice), width="stretch")
        st.subheader("国家排名")
        ranking_metric = metric_choice
        st.plotly_chart(top_countries(active_country_summary, ranking_metric), width="stretch")
        st.markdown(
            "<p class='quiet-note'>保护压力重点国家图只保留项目数达到一定分母的国家，并按急需保护占比排序，"
            "避免 1 个项目造成的 100% 极端值误导。条形文本为“急需保护项目/项目总数”，用于直接识别更需要关注的国家。</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(safeguarding_pressure_focus_chart(active_country_summary), width="stretch")
        st.subheader("原始数量与人口归一化差异")
        st.plotly_chart(rank_shift_chart(active_country_summary), width="stretch")

    with tab_time:
        st.subheader("时间演化")
        st.markdown(
            "<p class='quiet-note'>折线只连接有时间顺序的年份。名录类型颜色在全应用中保持一致，降低跨图阅读成本。</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(yearly_trend(filtered_elements), width="stretch")
        st.markdown(
            "<p class='quiet-note'>面积图显示每年不同名录类型的构成比例，帮助区分“总量变化”和“类型结构变化”。</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(type_share_timeline(filtered_elements), width="stretch")

    with tab_theme:
        st.subheader("主题与区域结构")
        theme_top_left, theme_top_right = st.columns([1, 1])
        with theme_top_left:
            st.plotly_chart(concept_chart(concepts, filtered_elements), width="stretch")
        with theme_top_right:
            st.plotly_chart(region_type_heatmap(filtered_country, countries), width="stretch")
        theme_bottom_left, theme_bottom_right = st.columns([1, 1])
        with theme_bottom_left:
            st.plotly_chart(concept_type_heatmap(concepts, filtered_elements), width="stretch")
        with theme_bottom_right:
            st.plotly_chart(region_urgent_share_chart(filtered_country, countries), width="stretch")
        st.markdown(
            "<p class='quiet-note'>共现热力图统计同一项目中同时出现的主题概念，补充单一频次排名无法呈现的主题组合关系。</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(concept_cooccurrence_heatmap(concepts, filtered_elements), width="stretch")

    with tab_network:
        st.subheader("跨国联合申报网络")
        focus_iso2 = None
        if focus_country != "全部国家":
            match = countries[countries["country_name"] == focus_country]
            if not match.empty:
                focus_iso2 = match.iloc[0]["iso2"]
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("参与国家", f"{network_stats['node_count']:,}")
        n2.metric("筛选后关系边", f"{network_stats['edge_count']:,}")
        n3.metric(
            "最强国家对",
            str(network_stats["strongest_pair"]),
            help=f"共享项目 {network_stats['strongest_weight']}",
        )
        n4.metric(
            "核心桥接国",
            str(network_stats["top_bridge"]),
            help=f"加权连接度 {network_stats['top_bridge_weighted_degree']}",
        )
        pair_col, centrality_col = st.columns([1, 1])
        with pair_col:
            st.plotly_chart(network_pair_chart(active_edges, min_weight=min_edge_weight), width="stretch")
        with centrality_col:
            st.plotly_chart(network_centrality_chart(active_edges, min_weight=min_edge_weight), width="stretch")
        st.plotly_chart(multinational_share_by_type(filtered_elements, filtered_country), width="stretch")
        st.markdown(
            "<p class='quiet-note'>联合申报规模分布展示项目由多少国家共同参与，避免只看国家对网络而忽略多国项目的规模差异。</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(collaboration_scale_distribution(filtered_country), width="stretch")
        st.plotly_chart(network_figure(active_edges, min_weight=min_edge_weight, focus_iso2=focus_iso2), width="stretch")

    with tab_country:
        st.subheader("国家详情")
        if focus_country == "全部国家":
            st.info("请在侧栏选择一个国家，查看该国的名录结构和代表性记录。")
        else:
            country_row = countries[countries["country_name"] == focus_country].iloc[0]
            display = build_country_detail_table(filtered_country, elements, country_row["iso2"])
            c1, c2, c3 = st.columns(3)
            c1.metric("项目数", int(display["非遗项目"].nunique()) if not display.empty else 0)
            c2.metric("人口分母", f"{country_row['population']:,.0f}" if pd.notna(country_row["population"]) else "缺失")
            c3.metric("所属区域", str(country_row["region"]))
            if display.empty:
                st.info("该国家在当前筛选条件下没有项目记录。")
            else:
                st.dataframe(display, hide_index=True, width="stretch")

    st.caption("数据来源：UNESCO DataHub Intangible Heritage List；World Bank Indicators API。")


if __name__ == "__main__":
    main()
