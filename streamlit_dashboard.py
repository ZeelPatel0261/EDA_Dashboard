"""
Streamlit Dashboard (Tabbed) - Accidents and Bikers

Run:
    pip install streamlit plotly pandas numpy
    streamlit run streamlit_dashboard_tabs.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly import graph_objects as go


# --------------- Data Loading & Utilities ---------------

def load_data() -> pd.DataFrame:
    path = "Accidents and Bikers.csv"
    df = pd.read_csv(path)
    # Drop rows with missing values similar to the EDA
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean


def safe_counts(s: pd.Series, label: str) -> pd.DataFrame:
    """Return a DataFrame with consistent columns [label, 'Count'] for value counts."""
    vc = s.value_counts(dropna=False)
    dfc = vc.rename_axis(label).reset_index(name="Count")
    return dfc


# --------------- Sidebar Filters ---------------

def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    severities = sorted(df["Severity"].unique())
    selected_severities = st.sidebar.multiselect("Select severities", options=severities, default=severities)
    genders = sorted(df["Gender"].unique())
    selected_genders = st.sidebar.multiselect("Select genders", options=genders, default=genders)
    min_year = int(df["Year"].min()); max_year = int(df["Year"].max())
    year_range = st.sidebar.slider("Year range", min_value=min_year, max_value=max_year, value=(min_year, max_year), step=1)
    light_conditions = sorted(df["Light_conditions"].dropna().unique())
    selected_light = st.sidebar.multiselect("Light conditions", options=light_conditions, default=light_conditions)

    filtered_df = df[
        (df["Severity"].isin(selected_severities)) &
        (df["Gender"].isin(selected_genders)) &
        (df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1]) &
        (df["Light_conditions"].isin(selected_light))
    ].copy()

    st.sidebar.markdown(f"**Filtered rows:** {len(filtered_df):,}")
    return filtered_df


# --------------- Tab: Overview ---------------

def tab_overview(df: pd.DataFrame):
    st.header("Overview")
    st.markdown("Use the sidebar to filter the dataset. This tab shows headline metrics and a sample preview.")
    st.write(f"**Rows:** {len(df):,}   **Columns:** {df.shape[1]}")
    with st.expander("Preview sample data"):
        st.dataframe(df.head())

    total_accidents = len(df)
    avg_casualties = df["Number_of_Casualties"].mean()
    avg_vehicles = df["Number_of_Vehicles"].mean()
    fatal_rate = (df["Severity"] == "Fatal").mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total accidents", f"{total_accidents:,}")
    c2.metric("Avg. casualties", f"{avg_casualties:.2f}")
    c3.metric("Avg. vehicles", f"{avg_vehicles:.2f}")
    c4.metric("Fatal accident rate", f"{fatal_rate:.2f}%")


# --------------- Tab: Distributions ---------------

def tab_distributions(df: pd.DataFrame):
    st.header("Distributions")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Accident severity")
        severity_counts = safe_counts(df["Severity"], "Severity")
        fig = px.bar(severity_counts, x="Severity", y="Count", color="Severity")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Rider gender")
        gender_counts = safe_counts(df["Gender"], "Gender")
        fig = px.bar(gender_counts, x="Gender", y="Count", color="Gender")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Age group distribution")
    age_counts = safe_counts(df["Age_Grp"], "Age group")
    fig = px.bar(age_counts, x="Age group", y="Count", color="Age group")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top road conditions")
    road_counts = safe_counts(df["Road_conditions"], "Road condition").nlargest(10, "Count")
    fig = px.bar(road_counts, x="Road condition", y="Count", color="Road condition")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Road type distribution")
    road_type_counts = safe_counts(df["Road_type"], "Road type")
    fig = px.bar(road_type_counts, x="Road type", y="Count", color="Road type")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Light conditions")
    light_counts = safe_counts(df["Light_conditions"], "Light condition")
    fig = px.bar(light_counts, x="Light condition", y="Count", color="Light condition")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribution of number of vehicles")
    fig = px.histogram(df, x="Number_of_Vehicles", nbins=int(df["Number_of_Vehicles"].max()))
    fig.update_xaxes(title="Number of vehicles"); fig.update_yaxes(title="Frequency")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribution of number of casualties")
    fig = px.histogram(df, x="Number_of_Casualties", nbins=int(df["Number_of_Casualties"].max()))
    fig.update_xaxes(title="Number of casualties"); fig.update_yaxes(title="Frequency")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Speed limits")
    fig = px.histogram(df, x="Speed_limit")
    fig.update_xaxes(title="Speed limit (mph)"); fig.update_yaxes(title="Frequency")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Accidents by day of week")
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day_counts = safe_counts(df["DayOfWeek"], "Day").set_index("Day").reindex(order).reset_index()
    day_counts["Count"] = day_counts["Count"].fillna(0)
    fig = px.bar(day_counts, x="Day", y="Count", color="Day")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Accidents by hour of day")
    hour_counts = safe_counts(df["Hour"], "Hour").sort_values("Hour")
    fig = px.bar(hour_counts, x="Hour", y="Count")
    st.plotly_chart(fig, use_container_width=True)


# --------------- Tab: Bivariate & Correlation ---------------

def tab_bivariate(df: pd.DataFrame):
    st.header("Bivariate Analysis & Correlation")

    st.subheader("Speed limit by severity")
    fig = px.box(df, x="Severity", y="Speed_limit", color="Severity")
    fig.update_yaxes(title="Speed limit (mph)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Number of vehicles by severity")
    fig = px.box(df, x="Severity", y="Number_of_Vehicles", color="Severity")
    fig.update_yaxes(title="Number of vehicles")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Severity by gender")
    sev_gender = pd.crosstab(df["Severity"], df["Gender"]).reset_index().melt(id_vars="Severity", var_name="Gender", value_name="Count")
    fig = px.bar(sev_gender, x="Severity", y="Count", color="Gender", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Road conditions vs severity (top 10)")
    top_rc = df["Road_conditions"].value_counts().nlargest(10).index
    pivot_rc = pd.crosstab(
        df[df["Road_conditions"].isin(top_rc)]["Road_conditions"],
        df[df["Road_conditions"].isin(top_rc)]["Severity"]
    ).reset_index().melt(id_vars="Road_conditions", var_name="Severity", value_name="Count")
    fig = px.density_heatmap(pivot_rc, x="Severity", y="Road_conditions", z="Count", color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Matrix (numeric variables)")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", labels=dict(color="Correlation"))
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough numeric columns found to compute a correlation matrix.")


# --------------- Tab: Time Series ---------------

def tab_time_series(df: pd.DataFrame):
    st.header("Time Series")

    st.subheader("Accidents per year")
    year_counts = df.groupby("Year").size().reset_index(name="Count")
    fig = px.line(year_counts, x="Year", y="Count", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Accidents by month")
    month_counts = df.groupby("Month").size().reindex(range(1, 13)).reset_index()
    month_counts.columns = ["Month", "Count"]
    month_counts["MonthName"] = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig = px.bar(month_counts, x="MonthName", y="Count", labels={"MonthName":"Month","Count":"Count"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily accident counts & 30-day moving average")
    try:
        df["Date_dt"] = pd.to_datetime(df["Date"], errors="coerce")
    except Exception:
        df["Date_dt"] = pd.to_datetime(df["Date"], errors="coerce")
    daily_counts = df.dropna(subset=["Date_dt"]).groupby("Date_dt").size().sort_index()
    if not daily_counts.empty:
        rolling = daily_counts.rolling(window=30).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_counts.index, y=daily_counts.values, mode="lines", name="Daily count", line=dict(color="lightgrey"), opacity=0.5))
        fig.add_trace(go.Scatter(x=rolling.index, y=rolling.values, mode="lines", name="30-day MA", line=dict(color="darkgreen")))
        fig.update_layout(xaxis_title="Date", yaxis_title="Accidents")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No valid dates available to plot the daily trend.")


# --------------- Tab: Notes / About ---------------

def tab_notes():
    st.header("Notes & Methodology")
    st.markdown(
        "- Includes: KPIs, univariate distributions, bivariate box plots & cross-tabs, correlation heatmap, and time series.\n"
        "- Filters: Severity, Gender, Year, and Light conditions apply to all tabs.\n"
        "- Interpretation: Correlation ≠ causation; distributions reflect exposure/context; boxplots show medians & quartiles."
    )


# --------------- Main App ---------------

def main():
    st.set_page_config(page_title="Accidents & Bikers Dashboard (Tabbed)", layout="wide")
    st.title("Accidents & Bikers - Tabbed Dashboard")

    df = load_data()
    df_filtered = sidebar_filters(df)

    tabs = st.tabs(["Overview", "Distributions", "Bivariate & Correlation", "Time Series", "Notes"])

    with tabs[0]:
        tab_overview(df_filtered)
    with tabs[1]:
        tab_distributions(df_filtered)
    with tabs[2]:
        tab_bivariate(df_filtered)
    with tabs[3]:
        tab_time_series(df_filtered)
    with tabs[4]:
        tab_notes()


if __name__ == "__main__":
    main()