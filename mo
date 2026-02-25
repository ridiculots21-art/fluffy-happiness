# Cell 1: build mapping of festive-related periods
from dateutil.relativedelta import relativedelta
from datetime import datetime
import polars as pl

# --- tweak this dict if you have other years / different months ---
lebaran_month_by_year = {
    2025: 3  # you said lebaran 2025 is month 3
}

def period_str_from_year_month(y, m):
    return f"{y} {m:02d}"

rows = []
for ly, lm in lebaran_month_by_year.items():
    leb = datetime(ly, lm, 1)
    # leb, m1 = leb -1 month, m2 = -2, m3 = -3
    m1 = leb + relativedelta(months=-1)
    m2 = leb + relativedelta(months=-2)
    m3 = leb + relativedelta(months=-3)
    rows.append({"lebaran_year": ly, "label": "leb", "periods": period_str_from_year_month(leb.year, leb.month)})
    rows.append({"lebaran_year": ly, "label": "1m",  "periods": period_str_from_year_month(m1.year, m1.month)})
    rows.append({"lebaran_year": ly, "label": "2m",  "periods": period_str_from_year_month(m2.year, m2.month)})
    rows.append({"lebaran_year": ly, "label": "3m",  "periods": period_str_from_year_month(m3.year, m3.month)})

lebaran_period_map = pl.DataFrame(rows).sort(["lebaran_year","label"])
lebaran_period_map

-------------------------------------------------------------------------------------------------------------------------------------------

# Cell 2: extract MA forecasts for the forecast horizon
start_period = "2025 01"
end_period   = "2025 06"

# ensure train_df contains 'key','periods','ma3','so_nw_ct'
forecast_ma_df = (
    train_df
    .select(["key","periods","ma3","so_nw_ct"])
    .filter((pl.col("periods") >= start_period) & (pl.col("periods") <= end_period))
    .sort(["key","periods"])
)
forecast_ma_df.head()

-------------------------------------------------------------------------------------------------------------------------------------------

# Cell 3: join label info onto forecast rows
# The mapping has years; we want to annotate each row with which lebaran_year it belongs to (if any)
# Join on 'periods' and keep lebaran_year & label (otherwise null)
forecast_with_label = forecast_ma_df.join(
    lebaran_period_map,
    on="periods",
    how="left"
).with_columns(
    pl.col("lebaran_year").cast(pl.Int64)  # can be null when period not related to a lebaran in the map
)

forecast_with_label.head(10)

-------------------------------------------------------------------------------------------------------------------------------------------

# Cell 4: join historical lebaran summary then pick appropriate festive avg per row
# ensure avg_summary_df column names match exactly:
# columns: 'lebaran_year', 'm3 avg', '2m avg', '1m avg', 'leb avg'

# rename avg columns to safe column names (no spaces) for easier referencing
avg_summary_df_clean = (
    avg_summary_df
    .with_columns([
        pl.col("m3 avg").alias("m3_avg"),
        pl.col("2m avg").alias("2m_avg"),
        pl.col("1m avg").alias("1m_avg"),
        pl.col("leb avg").alias("leb_avg")
    ])
    .select(["lebaran_year","m3_avg","2m_avg","1m_avg","leb_avg"])
)

merged = forecast_with_label.join(avg_summary_df_clean, on="lebaran_year", how="left")

# pick the correct festive average based on label
merged = merged.with_columns(
    pl.when(pl.col("label") == "3m").then(pl.col("m3_avg"))
      .when(pl.col("label") == "2m").then(pl.col("2m_avg"))
      .when(pl.col("label") == "1m").then(pl.col("1m_avg"))
      .when(pl.col("label") == "leb").then(pl.col("leb_avg"))
      .otherwise(None)
      .alias("festive_avg")
)

# quick check
merged.select(["key","periods","label","ma3","so_nw_ct","lebaran_year","festive_avg"]).head(12)

-------------------------------------------------------------------------------------------------------------------------------------------

# Cell 5: compute adjusted forecast
final_forecast = merged.with_columns(
    adjusted_forecast = pl.when(pl.col("festive_avg").is_not_null())
        .then((pl.col("ma3") + pl.col("festive_avg")) / 2.0)
        .otherwise(pl.col("ma3"))
).select(["key","periods","label","ma3","festive_avg","adjusted_forecast","so_nw_ct","lebaran_year"])

# sanity check
final_forecast.head(20)

-------------------------------------------------------------------------------------------------------------------------------------------

# Cell 6a: global aggregated plot (sum across all keys)
plot_df = (
    final_forecast
    .group_by("periods")
    .agg([
        pl.col("adjusted_forecast").sum().alias("forecast_sum"),
        pl.col("so_nw_ct").sum().alias("actual_sum")
    ])
    .sort("periods")
)

# convert to pandas for plotting convenience
plot_pd = plot_df.to_pandas()
plot_pd["periods_dt"] = pd.to_datetime(plot_pd["periods"].str.replace(" ", "-") + "-01", format="%Y-%m-%d")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,5))
plt.plot(plot_pd["periods_dt"], plot_pd["actual_sum"], marker="o", label="Actual (sum)")
plt.plot(plot_pd["periods_dt"], plot_pd["forecast_sum"], marker="o", label="Adjusted Forecast (sum)")
plt.xticks(plot_pd["periods_dt"], plot_pd["periods"], rotation=45)
plt.title("Total Actual vs Adjusted Forecast (2025 H1)")
plt.ylabel("Quantity (nw_ct)")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

-------------------------------------------------------------------------------------------------------------------------------------------

# Cell 6b: per-pareto group aggregated plot (if you have perform_df with 'pareto80_flag' and 'key')
# perform_df earlier had key and maybe pareto80_flag
if "pareto80_flag" in perform_df.columns:
    perf_small = perform_df.select(["key","pareto80_flag"])
    joined_pf = final_forecast.join(perf_small, on="key", how="left").with_columns(
        pl.col("pareto80_flag").fill_null(0).cast(pl.Int8)
    )
    grouped = joined_pf.group_by(["periods","pareto80_flag"]).agg([
        pl.col("adjusted_forecast").sum().alias("forecast_sum"),
        pl.col("so_nw_ct").sum().alias("actual_sum")
    ]).sort(["pareto80_flag","periods"])

    # convert to pandas and plot two lines per group
    gp_pd = grouped.to_pandas()
    gp_pd["periods_dt"] = pd.to_datetime(gp_pd["periods"].str.replace(" ", "-") + "-01", format="%Y-%m-%d")

    # simple loop plot
    for flag in sorted(gp_pd["pareto80_flag"].unique()):
        sub = gp_pd[gp_pd["pareto80_flag"] == flag].sort_values("periods_dt")
        plt.figure(figsize=(10,4))
        plt.plot(sub["periods_dt"], sub["actual_sum"], marker="o", label=f"Actual (pareto={flag})")
        plt.plot(sub["periods_dt"], sub["forecast_sum"], marker="o", label=f"Adj Forecast (pareto={flag})")
        plt.title(f"Actual vs Adjusted Forecast — pareto80_flag={flag}")
        plt.xticks(sub["periods_dt"], sub["periods"], rotation=45)
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()
else:
    print("perform_df does not have 'pareto80_flag' column — skip pareto plotting.")

-------------------------------------------------------------------------------------------------------------------------------------------

# Cell 7: per-key sample plots (top N keys by 2025 sales)
N = 4
top_keys = (
    final_forecast.filter(pl.col("periods").str.contains("2025"))
    .group_by("key").agg(pl.col("so_nw_ct").sum().alias("sales2025"))
    .sort("sales2025", descending=True)
    .head(N)["key"]
).to_list()

sample_df = final_forecast.filter(pl.col("key").is_in(top_keys)).sort(["key","periods"]).to_pandas()
sample_df["periods_dt"] = pd.to_datetime(sample_df["periods"].str.replace(" ", "-") + "-01", format="%Y-%m-%d")

import matplotlib.pyplot as plt
for k in top_keys:
    sub = sample_df[sample_df["key"] == k]
    plt.figure(figsize=(8,3.5))
    plt.plot(sub["periods_dt"], sub["so_nw_ct"], marker="o", label="actual")
    plt.plot(sub["periods_dt"], sub["adjusted_forecast"], marker="o", label="adj forecast")
    plt.title(f"Key: {k}")
    plt.xticks(sub["periods_dt"], sub["periods"], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

-------------------------------------------------------------------------------------------------------------------------------------------

# Cell 8: diagnostics
diag = final_forecast.with_columns(
    adjusted_flag = pl.when(pl.col("label").is_not_null()).then(1).otherwise(0),
    diff = (pl.col("adjusted_forecast") - pl.col("ma3"))
)
summary = {
    "n_rows": final_forecast.height,
    "n_adjusted_rows": diag.filter(pl.col("adjusted_flag")==1).height,
    "total_adjustment_sum": float(diag.select(pl.col("diff").sum()).to_numpy()[0][0]) if diag.height>0 else 0.0
}
summary, diag.select(["periods","label","ma3","festive_avg","adjusted_forecast","diff"]).filter(pl.col("label").is_not_null()).head(20)

-------------------------------------------------------------------------------------------------------------------------------------------

# CELL 4: build per-key festive averages from lebaran_base_df
# (lebaran_base_df was assembled previously in your lebaran section)
per_key_lebaran_avg = (
    lebaran_base_df
    .group_by("key")
    .agg([
        pl.col("m3_sales").mean().alias("m3_avg_key"),
        pl.col("m2_sales").mean().alias("m2_avg_key"),
        pl.col("m1_sales").mean().alias("m1_avg_key"),
        pl.col("lebaran_sales").mean().alias("leb_avg_key"),
        pl.count().alias("n_years_key")
    ])
)

print("Per-key lebaran avg sample:")
display(per_key_lebaran_avg.head(6).to_pandas())

-------------------------------------------------------------------------------------------------------------------------------------------

# CELL 5: join per-key festive averages onto forecast rows
merged = forecast_with_label.join(per_key_lebaran_avg, on="key", how="left")

# pick the correct festive avg for each label
merged = merged.with_columns(
    pl.when(pl.col("label") == "3m").then(pl.col("m3_avg_key"))
      .when(pl.col("label") == "2m").then(pl.col("m2_avg_key"))
      .when(pl.col("label") == "1m").then(pl.col("m1_avg_key"))
      .when(pl.col("label") == "leb").then(pl.col("leb_avg_key"))
      .otherwise(None)
      .alias("festive_avg")
)

# quick inspect
display(merged.select(["key","periods","label","lebaran_year","n_years_key","festive_avg","ma3","so_nw_ct"]).head(12).to_pandas())  

-------------------------------------------------------------------------------------------------------------------------------------------

# CELL 6: compute adjusted forecast (ma3 + festive_avg) / 2 when festive exists
WEIGHT_MA = 0.5
WEIGHT_FESTIVE = 0.5

final_forecast = merged.with_columns(
    adjusted_forecast = pl.when(pl.col("festive_avg").is_not_null() & pl.col("ma3").is_not_null())
        .then(pl.col("ma3") * WEIGHT_MA + pl.col("festive_avg") * WEIGHT_FESTIVE)
        .when(pl.col("ma3").is_not_null()).then(pl.col("ma3"))
        .otherwise(pl.col("festive_avg"))
)

# Inspect few rows (including ones that were flagged for lebaran months)
display(final_forecast.select(["key","periods","label","lebaran_year","n_years_key","festive_avg","ma3","adjusted_forecast","so_nw_ct"]).head(20).to_pandas())

-------------------------------------------------------------------------------------------------------------------------------------------

# CELL 7: compute errors and per-key mape averages for 2025-01 .. 2025-06
eval_df = final_forecast.with_columns([
    # MA errors
    (pl.col("ma3") - pl.col("so_nw_ct")).alias("re_ma"),
    (pl.col("ma3") - pl.col("so_nw_ct")).abs().alias("ae_ma"),
    # Adjusted errors
    (pl.col("adjusted_forecast") - pl.col("so_nw_ct")).alias("re_adj"),
    (pl.col("adjusted_forecast") - pl.col("so_nw_ct")).abs().alias("ae_adj"),
]).with_columns([
    # row-level mape, cap at 1
    pl.when(pl.col("so_nw_ct") > 0).then(pl.col("ae_ma") / pl.col("so_nw_ct")).otherwise(1).alias("mape_ma_raw"),
    pl.when(pl.col("so_nw_ct") > 0).then(pl.col("ae_adj") / pl.col("so_nw_ct")).otherwise(1).alias("mape_adj_raw")
]).with_columns([
    pl.when(pl.col("mape_ma_raw") > 1).then(1).otherwise(pl.col("mape_ma_raw")).alias("mape_ma"),
    pl.when(pl.col("mape_adj_raw") > 1).then(1).otherwise(pl.col("mape_adj_raw")).alias("mape_adj"),
])

# Keep only 2025 H1 rows
horizon_periods = ["2025 01","2025 02","2025 03","2025 04","2025 05","2025 06"]
eval_2025 = eval_df.filter(pl.col("periods").is_in(horizon_periods))

# Pivot MAPE per key (months -> columns), then compute horizontal average (per key)
mape_ma_pivot = eval_2025.pivot(on="periods", index="key", values="mape_ma", aggregate_function="sum", sort_columns=True)
mape_adj_pivot = eval_2025.pivot(on="periods", index="key", values="mape_adj", aggregate_function="sum", sort_columns=True)

# compute per-key averages across months (ignores missing months automatically)
mape_ma_pivot = mape_ma_pivot.with_columns(mape_ma_pivot.drop("key").mean_horizontal().alias("mape_avg_ma"))
mape_adj_pivot = mape_adj_pivot.with_columns(mape_adj_pivot.drop("key").mean_horizontal().alias("mape_avg_adj"))

# bring both together
mape_compare = mape_ma_pivot.select(["key","mape_avg_ma"]).join(
    mape_adj_pivot.select(["key","mape_avg_adj"]), on="key", how="outer"
).with_columns(
    pl.col("mape_avg_ma").fill_null(1.0),
    pl.col("mape_avg_adj").fill_null(1.0)
)

# attach pareto flag and zone (use perform_df which you already computed earlier)
if "pareto80_flag" in perform_df.columns:
    mape_compare = mape_compare.join(perform_df.select(["key","pareto80_flag"]), on="key", how="left").with_columns(
        pl.col("pareto80_flag").fill_null(0).cast(pl.Int8)
    )
else:
    mape_compare = mape_compare.with_columns(pl.lit(0).alias("pareto80_flag"))

# add zone
mape_compare = mape_compare.with_columns(pl.col("key").str.split_exact("_",1).list.get(0).alias("zone"))

print("Sample mape compare (per key):")
display(mape_compare.head(10).to_pandas())

-------------------------------------------------------------------------------------------------------------------------------------------

# CELL 8: global aggregated actual vs adjusted plot
plot_df = (
    final_forecast
    .group_by("periods")
    .agg([
        pl.col("adjusted_forecast").sum().alias("forecast_sum"),
        pl.col("ma3").sum().alias("ma_sum"),
        pl.col("so_nw_ct").sum().alias("actual_sum")
    ])
    .sort("periods")
).to_pandas()

plot_df["period_dt"] = pd.to_datetime(plot_df["periods"].str.replace(" ", "-") + "-01", format="%Y-%m-%d")

plt.figure(figsize=(10,5))
plt.plot(plot_df["period_dt"], plot_df["actual_sum"], marker="o", label="Actual (sum)")
plt.plot(plot_df["period_dt"], plot_df["ma_sum"], marker="o", label="MA3 (sum)")
plt.plot(plot_df["period_dt"], plot_df["forecast_sum"], marker="o", label="Adjusted (sum)")
plt.xticks(plot_df["period_dt"], plot_df["periods"], rotation=45)
plt.title("Total Actual vs MA3 vs Adjusted Forecast (2025 H1)")
plt.ylabel("Quantity (nw_ct)")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

-------------------------------------------------------------------------------------------------------------------------------------------

# CELL 9: per-zone aggregated comparison (pareto vs non-pareto) for MA and Adjusted
zone_summary = (
    mape_compare
    .group_by(["zone","pareto80_flag"])
    .agg([
        pl.col("mape_avg_ma").mean().alias("mape_ma_zone"),
        pl.col("mape_avg_adj").mean().alias("mape_adj_zone"),
        pl.count().alias("combination_count")
    ])
    .sort("zone")
)

zone_pd = zone_summary.to_pandas()

# Create pivot tables for plotting
pivot_ma = zone_pd.pivot(index="zone", columns="pareto80_flag", values="mape_ma_zone").fillna(0)
pivot_adj = zone_pd.pivot(index="zone", columns="pareto80_flag", values="mape_adj_zone").fillna(0)

# Ensure both flag columns present
for df in (pivot_ma, pivot_adj):
    if 0 not in df.columns: df[0] = 0
    if 1 not in df.columns: df[1] = 0
    # order columns: non-pareto (0) then pareto (1)
    df = df[[0,1]]

# Plot side-by-side subplots for direct comparison
fig, axes = plt.subplots(1,2, figsize=(16,5), sharey=True)
pivot_ma.sort_index().plot(kind="bar", ax=axes[0])
axes[0].set_title("Per-Zone Avg MAPE — MA3 (non-pareto=0 vs pareto=1)")
axes[0].set_ylabel("MAPE (proportion)")
axes[0].set_xlabel("Zone")
axes[0].legend(title="pareto80_flag", loc="upper right")

pivot_adj.sort_index().plot(kind="bar", ax=axes[1])
axes[1].set_title("Per-Zone Avg MAPE — Adjusted Forecast (non-pareto=0 vs pareto=1)")
axes[1].set_xlabel("Zone")
axes[1].legend(title="pareto80_flag", loc="upper right")

plt.tight_layout()
plt.show()

-------------------------------------------------------------------------------------------------------------------------------------------

# CELL 10: diagnostics
diag = final_forecast.with_columns(
    adjusted_flag = pl.when(pl.col("label").is_not_null() & pl.col("festive_avg").is_not_null()).then(1).otherwise(0),
    diff = (pl.col("adjusted_forecast") - pl.col("ma3"))
)

n_rows = diag.height
n_adjusted_rows = diag.filter(pl.col("adjusted_flag") == 1).height
total_adj = float(diag.select(pl.col("diff").sum()).to_numpy()[0][0]) if n_rows>0 else 0.0
avg_diff = float(diag.filter(pl.col("adjusted_flag")==1).select(pl.col("diff").mean()).to_numpy()[0][0]) if n_adjusted_rows>0 else 0.0

print(f"Total rows in horizon: {n_rows}")
print(f"Rows adjusted by festive avg: {n_adjusted_rows}")
print(f"Sum of adjustments (adjusted - ma3): {total_adj:.2f}")
print(f"Average adjustment among adjusted rows: {avg_diff:.2f}")

# sample adjusted rows
display(diag.select(["key","periods","label","n_years_key","festive_avg","ma3","adjusted_forecast","diff"]).filter(pl.col("adjusted_flag")==1).head(12).to_pandas())

-------------------------------------------------------------------------------------------------------------------------------------------



-------------------------------------------------------------------------------------------------------------------------------------------


                            
-------------------------------------------------------------------------------------------------------------------------------------------       


                            
-------------------------------------------------------------------------------------------------------------------------------------------



-------------------------------------------------------------------------------------------------------------------------------------------                            
