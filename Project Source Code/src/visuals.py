"""Reusable Plotly visualizations for the Streamlit app."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import math

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


TYPE_COLORS = {
    "RL": "#3A6EA5",
    "USL": "#C7503E",
    "Art18": "#6A9A67",
}

TYPE_LABELS = {
    "RL": "代表作名录",
    "USL": "急需保护名录",
    "Art18": "优秀实践名册",
}

TYPE_LABEL_ALIASES = {
    **TYPE_LABELS,
    "Representative List": "代表作名录",
    "Urgent Safeguarding": "急需保护名录",
    "Urgent Safeguarding List": "急需保护名录",
    "Good Practices": "优秀实践名册",
    "Register of Good Safeguarding Practices": "优秀实践名册",
}


def apply_theme(fig: go.Figure, height: int = 520) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family="Arial", size=13, color="#1F2933"),
        title=dict(x=0.5, xanchor="center", yanchor="top", font=dict(size=18)),
        margin=dict(l=32, r=28, t=68, b=38),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend_title_text="",
    )
    return fig


def empty_figure(title: str, message: str, height: int = 420) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_layout(title=title)
    return apply_theme(fig, height)


def coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def localize_type_label(value: object) -> object:
    if pd.isna(value):
        return value
    return TYPE_LABEL_ALIASES.get(str(value), value)


def filter_elements(elements: pd.DataFrame, years: tuple[int, int], list_types: list[str]) -> pd.DataFrame:
    start, end = years
    mask = elements["inscription_year"].between(start, end)
    if list_types:
        mask &= elements["type_acronym"].isin(list_types)
    return elements[mask].copy()


def filtered_country_rows(element_countries: pd.DataFrame, filtered_elements: pd.DataFrame) -> pd.DataFrame:
    return element_countries[element_countries["element_id"].isin(filtered_elements["element_id"])].copy()


def build_country_edges_from_rows(country_rows: pd.DataFrame) -> pd.DataFrame:
    edge_elements: dict[tuple[str, str], list[str]] = defaultdict(list)
    columns = ["source_iso2", "target_iso2", "weight", "shared_elements"]
    if country_rows.empty:
        return pd.DataFrame(columns=columns)

    for element_id, group in country_rows.groupby("element_id"):
        countries = sorted(
            {
                str(value).upper()
                for value in group["iso2"]
                if not pd.isna(value) and str(value).strip()
            }
        )
        for source, target in combinations(countries, 2):
            edge_elements[(source, target)].append(str(element_id))

    rows = [
        {
            "source_iso2": source,
            "target_iso2": target,
            "weight": len(element_ids),
            "shared_elements": ";".join(element_ids),
        }
        for (source, target), element_ids in edge_elements.items()
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["weight", "source_iso2", "target_iso2"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def network_summary(edges: pd.DataFrame) -> dict[str, object]:
    if edges.empty:
        return {
            "edge_count": 0,
            "node_count": 0,
            "strongest_pair": "无",
            "strongest_weight": 0,
            "top_bridge": "无",
            "top_bridge_weighted_degree": 0,
        }

    nodes = set(edges["source_iso2"]).union(set(edges["target_iso2"]))
    strongest = edges.nlargest(1, "weight").iloc[0]
    weighted_degree: dict[str, int] = {}
    for row in edges.itertuples():
        weighted_degree[row.source_iso2] = weighted_degree.get(row.source_iso2, 0) + int(row.weight)
        weighted_degree[row.target_iso2] = weighted_degree.get(row.target_iso2, 0) + int(row.weight)
    top_bridge, top_value = sorted(weighted_degree.items(), key=lambda item: item[1], reverse=True)[0]
    return {
        "edge_count": int(len(edges)),
        "node_count": int(len(nodes)),
        "strongest_pair": f"{strongest.source_iso2}-{strongest.target_iso2}",
        "strongest_weight": int(strongest.weight),
        "top_bridge": top_bridge,
        "top_bridge_weighted_degree": int(top_value),
    }


def recompute_country_summary(
    filtered_countries: pd.DataFrame,
    country_summary: pd.DataFrame,
) -> pd.DataFrame:
    base = country_summary[
        [
            "iso2",
            "iso3",
            "country_name",
            "region",
            "income_level",
            "population",
            "metric_missing",
        ]
    ].drop_duplicates("iso2")
    base = coerce_numeric_columns(base, ["population"])
    if filtered_countries.empty:
        base = base.copy()
        base["element_count"] = 0
        base["urgent_count"] = 0
        base["urgent_share"] = 0.0
        base["elements_per_million"] = float("nan")
        return base
    counts = (
        filtered_countries.groupby("iso2")
        .agg(
            element_count=("element_id", "nunique"),
            urgent_count=("type_acronym", lambda values: int((values == "USL").sum())),
        )
        .reset_index()
    )
    merged = base.merge(counts, on="iso2", how="left")
    merged["element_count"] = merged["element_count"].fillna(0).astype(int)
    merged["urgent_count"] = merged["urgent_count"].fillna(0).astype(int)
    merged["urgent_share"] = merged.apply(
        lambda row: row["urgent_count"] / row["element_count"] if row["element_count"] else 0,
        axis=1,
    )
    merged["elements_per_million"] = merged.apply(
        lambda row: row["element_count"] / (row["population"] / 1_000_000)
        if row["element_count"] and pd.notna(row["population"]) and row["population"] > 0
        else float("nan"),
        axis=1,
    )
    merged = coerce_numeric_columns(merged, ["urgent_share", "elements_per_million"])
    return merged


def build_country_detail_table(
    filtered_country: pd.DataFrame,
    elements: pd.DataFrame,
    iso2: str,
) -> pd.DataFrame:
    country_rows = filtered_country[filtered_country["iso2"] == iso2].copy()
    if country_rows.empty:
        return pd.DataFrame(columns=["非遗项目", "名录类型", "入选年份", "UNESCO 链接"])

    metadata_columns = ["element_id", "type_label", "http_url_en"]
    optional_columns = ["title_en", "inscription_year"]
    available_columns = metadata_columns + [column for column in optional_columns if column in elements.columns]
    element_metadata = elements[available_columns].drop_duplicates("element_id")
    merged = country_rows.merge(element_metadata, on="element_id", how="left", suffixes=("", "_element"))

    for column in optional_columns:
        fallback = f"{column}_element"
        if column not in merged.columns and fallback in merged.columns:
            merged[column] = merged[fallback]
        elif fallback in merged.columns:
            merged[column] = merged[column].fillna(merged[fallback])

    merged["type_label"] = merged.get("type_label", merged["type_acronym"].map(TYPE_LABELS))
    merged["type_label"] = merged["type_label"].fillna(merged["type_acronym"].map(TYPE_LABELS))
    merged["type_label"] = merged["type_label"].map(localize_type_label)

    display = merged[["title_en", "type_label", "inscription_year", "http_url_en"]].drop_duplicates()
    display = display.sort_values(["inscription_year", "title_en"], ascending=[False, True])
    return display.rename(
        columns={
            "title_en": "非遗项目",
            "type_label": "名录类型",
            "inscription_year": "入选年份",
            "http_url_en": "UNESCO 链接",
        }
    )


def world_map(summary: pd.DataFrame, metric: str) -> go.Figure:
    labels = {
        "elements_per_million": "每百万人项目数",
        "element_count": "国家项目数量",
        "urgent_share": "急需保护占比",
    }
    data = summary[summary["element_count"] > 0].copy()
    data = coerce_numeric_columns(data, ["element_count", "urgent_count", "urgent_share", "elements_per_million"])
    if metric == "elements_per_million":
        data = data[data["metric_missing"] == False]  # noqa: E712
    fig = px.choropleth(
        data,
        locations="iso3",
        color=metric,
        hover_name="country_name",
        hover_data={
            "iso3": False,
            "element_count": ":,.0f",
            "urgent_count": ":,.0f",
            "urgent_share": ":.1%",
            "elements_per_million": ":.2f",
        },
        color_continuous_scale="YlGnBu" if metric != "urgent_share" else "OrRd",
        labels={metric: labels[metric]},
        title=f"世界地图：{labels[metric]}",
    )
    fig.update_geos(showframe=False, showcoastlines=True, coastlinecolor="#CBD2D9", projection_type="natural earth")
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=labels[metric],
            thickness=14,
            len=0.74,
            tickformat=".2f" if metric == "elements_per_million" else ".0%" if metric == "urgent_share" else ",.0f",
        )
    )
    return apply_theme(fig, 700)


def yearly_trend(elements: pd.DataFrame) -> go.Figure:
    yearly = (
        elements.groupby(["inscription_year", "type_acronym"])
        .agg(element_count=("element_id", "nunique"))
        .reset_index()
    )
    fig = px.line(
        yearly,
        x="inscription_year",
        y="element_count",
        color="type_acronym",
        markers=True,
        color_discrete_map=TYPE_COLORS,
        category_orders={"type_acronym": ["RL", "USL", "Art18"]},
        labels={"inscription_year": "年份", "element_count": "项目数", "type_acronym": "名录类型"},
        title="不同名录类型的入选时间演化",
    )
    fig.update_xaxes(dtick=2)
    fig.update_yaxes(rangemode="tozero")
    return apply_theme(fig, 420)


def type_share_timeline(elements: pd.DataFrame) -> go.Figure:
    if elements.empty:
        return empty_figure("年度名录构成", "当前筛选条件下没有可计算的年度构成。")
    yearly = (
        elements.groupby(["inscription_year", "type_acronym"])
        .agg(element_count=("element_id", "nunique"))
        .reset_index()
    )
    totals = yearly.groupby("inscription_year")["element_count"].transform("sum")
    yearly["type_share"] = yearly["element_count"] / totals
    yearly["type_label"] = yearly["type_acronym"].map(TYPE_LABELS).fillna(yearly["type_acronym"])
    fig = px.area(
        yearly,
        x="inscription_year",
        y="type_share",
        color="type_label",
        color_discrete_sequence=[TYPE_COLORS["RL"], TYPE_COLORS["USL"], TYPE_COLORS["Art18"]],
        category_orders={"type_label": [TYPE_LABELS["RL"], TYPE_LABELS["USL"], TYPE_LABELS["Art18"]]},
        labels={"inscription_year": "年份", "type_share": "年度项目占比", "type_label": "名录类型"},
        title="年度名录构成（按项目占比）",
    )
    fig.update_xaxes(dtick=2)
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return apply_theme(fig, 420)


def top_countries(summary: pd.DataFrame, metric: str, n: int = 15) -> go.Figure:
    data = summary[summary["element_count"] > 0].copy()
    data = coerce_numeric_columns(data, ["element_count", "urgent_share", "elements_per_million"])
    if metric == "elements_per_million":
        data = data[data["metric_missing"] == False]  # noqa: E712
    data = data.nlargest(n, metric).sort_values(metric)
    fig = px.bar(
        data,
        x=metric,
        y="country_name",
        orientation="h",
        color="urgent_share",
        color_continuous_scale="OrRd",
        text=metric,
        labels={
            metric: "每百万人项目数" if metric == "elements_per_million" else "项目数",
            "country_name": "国家",
            "urgent_share": "急需保护占比",
        },
        title="当前指标下的国家排名",
    )
    texttemplate = "%{text:.0%}" if metric == "urgent_share" else "%{text:.2f}" if metric == "elements_per_million" else "%{text:.0f}"
    fig.update_traces(texttemplate=texttemplate, textposition="outside", cliponaxis=False)
    fig.update_xaxes(rangemode="tozero")
    if metric == "urgent_share":
        fig.update_xaxes(tickformat=".0%")
    return apply_theme(fig, 520)


def concept_chart(concepts: pd.DataFrame, filtered_elements: pd.DataFrame, n: int = 18) -> go.Figure:
    filtered = concepts[concepts["element_id"].isin(filtered_elements["element_id"])]
    counts = filtered["concept_name"].value_counts().head(n).sort_values()
    fig = px.bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        text=counts.values,
        labels={"x": "项目数", "y": "主题概念"},
        title="当前筛选下的核心主题结构",
    )
    fig.update_traces(marker_color="#3A6EA5", textposition="outside", cliponaxis=False)
    fig.update_xaxes(rangemode="tozero")
    return apply_theme(fig, 500)


def region_type_heatmap(filtered_country: pd.DataFrame, country_summary: pd.DataFrame) -> go.Figure:
    joined = filtered_country.merge(country_summary[["iso2", "region"]], on="iso2", how="left")
    if joined.empty:
        matrix = pd.DataFrame([[0]], index=["无数据"], columns=["无数据"])
    else:
        matrix = (
            joined.groupby(["region", "type_acronym"])["element_id"]
            .nunique()
            .reset_index()
            .pivot(index="region", columns="type_acronym", values="element_id")
            .fillna(0)
        )
    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={"x": "名录类型", "y": "World Bank 区域", "color": "项目数"},
        title="区域与名录类型矩阵",
    )
    return apply_theme(fig, 500)


def safeguarding_pressure_focus_chart(summary: pd.DataFrame, min_projects: int = 3, n: int = 14) -> go.Figure:
    data = summary.copy()
    if "metric_missing" in data.columns:
        data = data[~data["metric_missing"].astype(str).str.lower().eq("true")]
    data = coerce_numeric_columns(
        data,
        ["element_count", "urgent_count", "urgent_share", "elements_per_million", "population"],
    )
    data = data[(data["element_count"] >= min_projects) & data["urgent_share"].notna()].copy()
    if data.empty:
        return empty_figure(
            "保护压力重点国家",
            f"当前筛选条件下没有项目数不少于 {min_projects} 的国家。",
        )

    data = data.nlargest(n, ["urgent_share", "urgent_count", "element_count"]).sort_values(
        ["urgent_share", "urgent_count", "element_count"],
        ascending=[False, False, False],
    )
    data["urgent_label"] = data.apply(
        lambda row: f"{int(row['urgent_count'])}/{int(row['element_count'])}",
        axis=1,
    )
    fig = px.bar(
        data,
        x="country_name",
        y="urgent_share",
        text="urgent_label",
        hover_data={
            "country_name": False,
            "region": True,
            "urgent_count": ":,.0f",
            "element_count": ":,.0f",
            "elements_per_million": ":.2f",
            "population": ":,.0f",
            "urgent_share": ":.1%",
        },
        labels={
            "country_name": "国家",
            "urgent_share": "急需保护占比",
            "urgent_count": "急需保护项目",
            "element_count": "项目数",
            "elements_per_million": "每百万人项目数",
            "population": "人口",
            "region": "World Bank 区域",
        },
        title=f"保护压力重点国家（项目数不少于 {min_projects}）",
    )
    fig.update_traces(marker_color="#C7503E", textposition="outside", cliponaxis=False)
    fig.update_xaxes(tickangle=-30)
    fig.update_yaxes(tickformat=".0%", rangemode="tozero", range=[0, 1.08])
    return apply_theme(fig, 520)


def regional_visibility_scatter(summary: pd.DataFrame) -> go.Figure:
    return safeguarding_pressure_focus_chart(summary)


def network_pair_chart(edges: pd.DataFrame, min_weight: int = 1, n: int = 12) -> go.Figure:
    data = edges[edges["weight"] >= min_weight].nlargest(n, "weight").copy()
    if data.empty:
        return empty_figure("强合作国家对", "当前筛选条件下没有达到阈值的跨国关系。")
    data["pair"] = data["source_iso2"] + "-" + data["target_iso2"]
    data = data.sort_values("weight")
    fig = px.bar(
        data,
        x="weight",
        y="pair",
        orientation="h",
        text="weight",
        labels={"weight": "共享项目数", "pair": "国家对"},
        title="强合作国家对（共享申报项目数）",
    )
    fig.update_traces(marker_color="#3A6EA5", textposition="outside", cliponaxis=False)
    fig.update_xaxes(rangemode="tozero")
    return apply_theme(fig, 420)


def network_centrality_chart(edges: pd.DataFrame, min_weight: int = 1, n: int = 12) -> go.Figure:
    data = edges[edges["weight"] >= min_weight].copy()
    if data.empty:
        return empty_figure("国家加权连接度", "当前筛选条件下没有可计算的网络中心性。")
    weighted_degree: dict[str, int] = {}
    for row in data.itertuples():
        weighted_degree[row.source_iso2] = weighted_degree.get(row.source_iso2, 0) + int(row.weight)
        weighted_degree[row.target_iso2] = weighted_degree.get(row.target_iso2, 0) + int(row.weight)
    centrality = (
        pd.DataFrame({"iso2": list(weighted_degree), "weighted_degree": list(weighted_degree.values())})
        .nlargest(n, "weighted_degree")
        .sort_values("weighted_degree")
    )
    fig = px.bar(
        centrality,
        x="weighted_degree",
        y="iso2",
        orientation="h",
        text="weighted_degree",
        labels={"weighted_degree": "加权连接度", "iso2": "国家"},
        title="核心桥接国家（加权连接度）",
    )
    fig.update_traces(marker_color="#6A9A67", textposition="outside", cliponaxis=False)
    fig.update_xaxes(rangemode="tozero")
    return apply_theme(fig, 420)


def multinational_share_by_type(elements: pd.DataFrame, element_countries: pd.DataFrame) -> go.Figure:
    if elements.empty or element_countries.empty:
        return empty_figure("跨国项目占比（按名录类型）", "当前筛选条件下没有项目。")
    country_counts = element_countries.groupby("element_id")["iso2"].nunique().rename("country_count")
    data = elements[["element_id", "type_acronym"]].drop_duplicates("element_id").merge(
        country_counts,
        on="element_id",
        how="left",
    )
    data["country_count"] = data["country_count"].fillna(0)
    data["is_multinational"] = data["country_count"] > 1
    summary = (
        data.groupby("type_acronym")
        .agg(element_count=("element_id", "nunique"), multinational_count=("is_multinational", "sum"))
        .reset_index()
    )
    summary["multinational_share"] = summary.apply(
        lambda row: row["multinational_count"] / row["element_count"] if row["element_count"] else 0,
        axis=1,
    )
    summary["type_label"] = summary["type_acronym"].map(TYPE_LABELS).fillna(summary["type_acronym"])
    fig = px.bar(
        summary,
        x="type_label",
        y="multinational_share",
        text="multinational_share",
        hover_data={"element_count": ":,.0f", "multinational_count": ":,.0f", "type_label": False},
        labels={"type_label": "", "multinational_share": "跨国项目占比"},
        title="跨国项目占比（按名录类型）",
    )
    fig.update_traces(marker_color="#3A6EA5", texttemplate="%{text:.1%}", textposition="outside", cliponaxis=False)
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return apply_theme(fig, 420)


def region_urgent_share_chart(filtered_country: pd.DataFrame, countries: pd.DataFrame) -> go.Figure:
    if filtered_country.empty:
        return empty_figure("区域急需保护占比", "当前筛选条件下没有国家-项目记录。")
    joined = filtered_country.merge(countries[["iso2", "region"]], on="iso2", how="left")
    joined["region"] = joined["region"].replace("", "未映射区域").fillna("未映射区域")
    dedup = joined.drop_duplicates(["region", "element_id"]).copy()
    dedup["is_urgent"] = dedup["type_acronym"] == "USL"
    summary = (
        dedup.groupby("region")
        .agg(element_count=("element_id", "nunique"), urgent_count=("is_urgent", "sum"))
        .reset_index()
    )
    summary["urgent_share"] = summary.apply(
        lambda row: row["urgent_count"] / row["element_count"] if row["element_count"] else 0,
        axis=1,
    )
    summary = summary.sort_values("urgent_share")
    fig = px.bar(
        summary,
        x="urgent_share",
        y="region",
        orientation="h",
        text="urgent_share",
        hover_data={"element_count": ":,.0f", "urgent_count": ":,.0f", "region": False},
        labels={"urgent_share": "急需保护占比", "region": "World Bank 区域"},
        title="区域急需保护占比（含项目分母）",
    )
    fig.update_traces(marker_color="#C7503E", texttemplate="%{text:.1%}", textposition="outside", cliponaxis=False)
    fig.update_xaxes(tickformat=".0%", rangemode="tozero")
    return apply_theme(fig, 500)


def rank_shift_chart(summary: pd.DataFrame, n: int = 15) -> go.Figure:
    data = summary[summary["element_count"] > 0].copy()
    if "metric_missing" in data.columns:
        data = data[~data["metric_missing"].astype(str).str.lower().eq("true")]
    data = coerce_numeric_columns(data, ["element_count", "elements_per_million"])
    data = data.dropna(subset=["elements_per_million"])
    if data.empty:
        return empty_figure("原始数量与人口归一化排名差异", "当前筛选条件下没有可比较的归一化数据。")
    data["raw_rank"] = data["element_count"].rank(method="min", ascending=False)
    data["normalized_rank"] = data["elements_per_million"].rank(method="min", ascending=False)
    data["rank_shift"] = data["raw_rank"] - data["normalized_rank"]
    data = data.reindex(data["rank_shift"].abs().sort_values(ascending=False).index).head(n)
    data = data.sort_values("rank_shift")
    fig = px.bar(
        data,
        x="rank_shift",
        y="country_name",
        orientation="h",
        text="rank_shift",
        labels={"rank_shift": "原始排名 - 每百万人排名", "country_name": "国家"},
        title="原始数量与人口归一化的排名差异",
    )
    fig.update_traces(marker_color="#6A9A67", texttemplate="%{text:.0f}", textposition="outside", cliponaxis=False)
    fig.add_vline(x=0, line_color="#52606D", line_width=1)
    return apply_theme(fig, 520)


def concept_type_heatmap(concepts: pd.DataFrame, filtered_elements: pd.DataFrame, n: int = 14) -> go.Figure:
    filtered = concepts[concepts["element_id"].isin(filtered_elements["element_id"])].copy()
    if filtered.empty:
        return empty_figure("主题-名录类型结构", "当前筛选条件下没有主题概念记录。")
    top_concepts = filtered["concept_name"].value_counts().head(n).index
    filtered = filtered[filtered["concept_name"].isin(top_concepts)]
    matrix = (
        filtered.groupby(["concept_name", "type_acronym"])["element_id"]
        .nunique()
        .reset_index()
        .pivot(index="concept_name", columns="type_acronym", values="element_id")
        .fillna(0)
    )
    matrix = matrix.reindex(columns=["RL", "USL", "Art18"]).fillna(0)
    matrix = matrix.loc[top_concepts]
    matrix = matrix.rename(columns=TYPE_LABELS)
    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={"x": "名录类型", "y": "主题概念", "color": "项目数"},
        title="主题-名录类型结构",
    )
    return apply_theme(fig, 500)


def concept_cooccurrence_heatmap(concepts: pd.DataFrame, filtered_elements: pd.DataFrame, n: int = 12) -> go.Figure:
    filtered = concepts[concepts["element_id"].isin(filtered_elements["element_id"])].copy()
    if filtered.empty:
        return empty_figure("主题共现结构", "当前筛选条件下没有主题概念记录。")
    top_concepts = filtered["concept_name"].value_counts().head(n).index.tolist()
    filtered = filtered[filtered["concept_name"].isin(top_concepts)]
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for _, group in filtered.groupby("element_id"):
        labels = sorted(set(group["concept_name"].dropna()))
        for source, target in combinations(labels, 2):
            pair_counts[(source, target)] += 1
    matrix = pd.DataFrame(0, index=top_concepts, columns=top_concepts, dtype=int)
    for (source, target), value in pair_counts.items():
        matrix.loc[source, target] = value
        matrix.loc[target, source] = value
    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="YlGnBu",
        labels={"x": "主题概念", "y": "主题概念", "color": "共现项目数"},
        title="主题共现结构（同一项目内）",
    )
    return apply_theme(fig, 600)


def collaboration_scale_distribution(filtered_country: pd.DataFrame) -> go.Figure:
    if filtered_country.empty:
        return empty_figure("联合申报规模分布", "当前筛选条件下没有国家-项目记录。")
    country_counts = filtered_country.groupby("element_id")["iso2"].nunique().rename("country_count")
    distribution = country_counts.value_counts().sort_index().rename_axis("country_count").reset_index(name="element_count")
    distribution["scale_label"] = distribution["country_count"].apply(lambda value: "1" if value == 1 else f"{value}")
    fig = px.bar(
        distribution,
        x="scale_label",
        y="element_count",
        text="element_count",
        labels={"scale_label": "参与国家数", "element_count": "项目数"},
        title="联合申报规模分布",
    )
    fig.update_traces(marker_color="#3A6EA5", textposition="outside", cliponaxis=False)
    fig.update_xaxes(type="category")
    fig.update_yaxes(rangemode="tozero")
    return apply_theme(fig, 420)


def network_figure(edges: pd.DataFrame, min_weight: int = 2, max_edges: int = 60, focus_iso2: str | None = None) -> go.Figure:
    filtered = edges[edges["weight"] >= min_weight].nlargest(max_edges, "weight").copy()
    if focus_iso2:
        focus_iso2 = focus_iso2.upper()
        filtered = filtered[(filtered["source_iso2"] == focus_iso2) | (filtered["target_iso2"] == focus_iso2)]

    graph = nx.Graph()
    for row in filtered.itertuples():
        graph.add_edge(row.source_iso2, row.target_iso2, weight=row.weight)

    if graph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="当前筛选条件下没有跨国联合申报关系。", showarrow=False)
        return apply_theme(fig, 480)

    pos = nx.spring_layout(graph, seed=11, weight="weight", k=1.7 / math.sqrt(graph.number_of_nodes()), iterations=120)
    edge_x, edge_y = [], []
    edge_mid_x, edge_mid_y, edge_text, edge_customdata = [], [], [], []
    edge_widths = []
    max_weight = max(nx.get_edge_attributes(graph, "weight").values())
    for source, target, attrs in graph.edges(data=True):
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_mid_x.append((x0 + x1) / 2)
        edge_mid_y.append((y0 + y1) / 2)
        edge_text.append(f"{source}-{target}")
        edge_customdata.append(attrs["weight"])
        edge_widths.append(0.8 + 2.4 * attrs["weight"] / max_weight)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=max(1.2, sum(edge_widths) / len(edge_widths)), color="rgba(58, 110, 165, 0.28)"),
        hoverinfo="skip",
        showlegend=False,
    )
    edge_hover_trace = go.Scatter(
        x=edge_mid_x,
        y=edge_mid_y,
        text=edge_text,
        customdata=edge_customdata,
        mode="markers",
        marker=dict(size=12, color="rgba(58, 110, 165, 0.01)"),
        hovertemplate="<b>%{text}</b><br>共享项目：%{customdata}<extra></extra>",
        showlegend=False,
    )
    degrees = dict(graph.degree(weight="weight"))
    top_nodes = {node for node, _ in sorted(degrees.items(), key=lambda item: item[1], reverse=True)[:14]}
    if focus_iso2:
        top_nodes.add(focus_iso2)
    node_trace = go.Scatter(
        x=[pos[node][0] for node in graph.nodes()],
        y=[pos[node][1] for node in graph.nodes()],
        text=[node if node in top_nodes else "" for node in graph.nodes()],
        customdata=[[node, degrees[node]] for node in graph.nodes()],
        mode="markers+text",
        textposition="top center",
        marker=dict(
            size=[8 + math.sqrt(degrees[node]) * 5 for node in graph.nodes()],
            color=["#D97706" if focus_iso2 and node == focus_iso2 else "#3A6EA5" for node in graph.nodes()],
            line=dict(color="white", width=1),
        ),
        hovertemplate="<b>%{customdata[0]}</b><br>加权连接度：%{customdata[1]:.0f}<extra></extra>",
        showlegend=False,
    )
    fig = go.Figure([edge_trace, edge_hover_trace, node_trace])
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(title="跨国联合申报网络（筛选后）", showlegend=False)
    return apply_theme(fig, 620)
