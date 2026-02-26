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

# Cell 4: festive averages per key with separate columns

festive_avg_df = (
    df_pareto
    .join(
        lebaran_period_map,
        on="periods",
        how="inner"
    )
    .group_by(["key", "label"])
    .agg(
        pl.col("so_nw_ct").mean().alias("festive_avg")
    )
    .pivot(
        index="key",
        on="label",
        values="festive_avg",
        aggregate_function="first"
    )
    .rename({
        "3m": "festive_avg_m3",
        "2m": "festive_avg_m2",
        "1m": "festive_avg_m1",
        "leb": "festive_avg_leb"
    })
    .sort("key")
)

print(festive_avg_df.shape)
festive_avg_df.head(10)

-------------------------------------------------------------------------------------------------------------------------------------------

# Cell 5: convert festive averages into ratios

# compute historical mean sales per key
key_mean_df = (
    df_pareto
    .group_by("key")
    .agg(
        pl.col("so_nw_ct").mean().alias("key_mean")
    )
)

# join mean and convert to ratios
festive_ratio_df = (
    festive_avg_df
    .join(key_mean_df, on="key", how="left")
    .with_columns([
        (pl.col("festive_avg_m3") / pl.col("key_mean")).alias("ratio_m3"),
        (pl.col("festive_avg_m2") / pl.col("key_mean")).alias("ratio_m2"),
        (pl.col("festive_avg_m1") / pl.col("key_mean")).alias("ratio_m1"),
        (pl.col("festive_avg_leb") / pl.col("key_mean")).alias("ratio_leb"),
    ])
    .select([
        "key",
        "ratio_m3",
        "ratio_m2",
        "ratio_m1",
        "ratio_leb"
    ])
)

print(festive_ratio_df.shape)
festive_ratio_df.head(10)

-------------------------------------------------------------------------------------------------------------------------------------------

# Cell 6: reshape festive ratios to long format (key + label)

festive_ratio_long = (
    festive_ratio_df
    .with_columns([
        pl.col("ratio_m3").alias("3m"),
        pl.col("ratio_m2").alias("2m"),
        pl.col("ratio_m1").alias("1m"),
        pl.col("ratio_leb").alias("leb"),
    ])
    .select(["key", "3m", "2m", "1m", "leb"])
    .melt(
        id_vars="key",
        variable_name="label",
        value_name="festive_ratio"
    )
    .filter(~pl.col("festive_ratio").is_null())
    .sort(["key", "label"])
)

print(festive_ratio_long.shape)
festive_ratio_long.head(10)

-------------------------------------------------------------------------------------------------------------------------------------------

# Cell 7: attach festive ratio to forecast rows (by key + label)

forecast_with_ratio = (
    forecast_with_label
    .join(
        festive_ratio_long,
        on=["key", "label"],
        how="left"   # important: keep all forecast rows
    )
    .sort(["key", "periods"])
)

print(forecast_with_ratio.shape)
forecast_with_ratio.head(10)

forecast_with_ratio.filter(pl.col("key") == "some_key_here")

-------------------------------------------------------------------------------------------------------------------------------------------

# Cell 8: clean festive ratio and apply to MA forecast

forecast_adjusted = (
    forecast_with_ratio
    # Step 1: convert zero ratios to null
    .with_columns(
        pl.when(pl.col("festive_ratio") == 0)
          .then(None)
          .otherwise(pl.col("festive_ratio"))
          .alias("festive_ratio_clean")
    )
    # Step 2: create effective ratio (default = 1 when null)
    .with_columns(
        pl.when(pl.col("festive_ratio_clean").is_null())
          .then(1)
          .otherwise(pl.col("festive_ratio_clean"))
          .alias("effective_ratio")
    )
    # Step 3: apply ratio to MA forecast
    .with_columns(
        (pl.col("ma3") * pl.col("effective_ratio"))
        .alias("festive_adjusted_forecast")
    )
    .sort(["key", "periods"])
)

print(forecast_adjusted.shape)
forecast_adjusted.select([
    "key", "periods", "label",
    "ma3", "festive_ratio", 
    "effective_ratio",
    "festive_adjusted_forecast",
    "so_nw_ct"
]).head(12)

-------------------------------------------------------------------------------------------------------------------------------------------

# Cell 9: final smoothed seasonal forecast

forecast_final = (
    forecast_adjusted
    .with_columns(
        (
            pl.col("ma3") * ((1 + pl.col("effective_ratio")) / 2)
        ).alias("final_forecast")
    )
    .sort(["key", "periods"])
)

print(forecast_final.shape)

forecast_final.select([
    "key",
    "periods",
    "label",
    "ma3",
    "effective_ratio",
    "final_forecast",
    "so_nw_ct"
]).head(12)

-------------------------------------------------------------------------------------------------------------------------------------------

# Cell 10: Evaluate MA3 vs Final Forecast (MAPE comparison)

evaluation_df = (
    forecast_final
    # Absolute Errors
    .with_columns([
        (pl.col("ma3") - pl.col("so_nw_ct")).abs().alias("ae_ma"),
        (pl.col("final_forecast") - pl.col("so_nw_ct")).abs().alias("ae_final")
    ])
    # MAPE
    .with_columns([
        pl.when(pl.col("so_nw_ct") > 0)
          .then(pl.col("ae_ma") / pl.col("so_nw_ct"))
          .otherwise(1)
          .alias("mape_ma"),

        pl.when(pl.col("so_nw_ct") > 0)
          .then(pl.col("ae_final") / pl.col("so_nw_ct"))
          .otherwise(1)
          .alias("mape_final")
    ])
    # Cap at 1 (100%)
    .with_columns([
        pl.when(pl.col("mape_ma") > 1).then(1).otherwise(pl.col("mape_ma")).alias("mape_ma"),
        pl.when(pl.col("mape_final") > 1).then(1).otherwise(pl.col("mape_final")).alias("mape_final"),
    ])
)

# Aggregate comparison
comparison = (
    evaluation_df
    .select([
        pl.col("mape_ma").mean().round(4).alias("avg_mape_ma"),
        pl.col("mape_final").mean().round(4).alias("avg_mape_final")
    ])
)

comparison

-------------------------------------------------------------------------------------------------------------------------------------------

# Cell 11: Line plot comparison for one key

import matplotlib.pyplot as plt
import pandas as pd

# 🔹 Change this to any key you want to inspect
selected_key = forecast_final["key"].unique()[0]

plot_df = (
    forecast_final
    .filter(pl.col("key") == selected_key)
    .select([
        "periods",
        "ma3",
        "final_forecast",
        "so_nw_ct"
    ])
    .with_columns(
        pl.col("periods").str.strptime(pl.Date, "%Y %m").alias("period_dt")
    )
    .sort("period_dt")
    .to_pandas()
)

plt.figure(figsize=(10,5))

plt.plot(plot_df["period_dt"], plot_df["so_nw_ct"], marker="o", label="Actual")
plt.plot(plot_df["period_dt"], plot_df["ma3"], marker="o", label="MA3 Forecast")
plt.plot(plot_df["period_dt"], plot_df["final_forecast"], marker="o", linestyle="--", label="Festive Adjusted Forecast")

plt.title(f"Forecast Comparison — Key: {selected_key}")
plt.xlabel("Period")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# Cell 11: Aggregate all keys and plot overall performance

import matplotlib.pyplot as plt
import pandas as pd

# Aggregate by period
agg_plot_df = (
    forecast_final
    .group_by("periods")
    .agg([
        pl.col("so_nw_ct").sum().alias("actual_total"),
        pl.col("ma3").sum().alias("ma_total"),
        pl.col("final_forecast").sum().alias("final_total"),
    ])
    .with_columns(
        pl.col("periods").str.strptime(pl.Date, "%Y %m").alias("period_dt")
    )
    .sort("period_dt")
    .to_pandas()
)

plt.figure(figsize=(10,5))

plt.plot(agg_plot_df["period_dt"], agg_plot_df["actual_total"], marker="o", label="Actual Total")
plt.plot(agg_plot_df["period_dt"], agg_plot_df["ma_total"], marker="o", label="MA3 Total")
plt.plot(agg_plot_df["period_dt"], agg_plot_df["final_total"], marker="o", linestyle="--", label="Festive Adjusted Total")

plt.title("Overall Forecast Comparison (All Keys Combined)")
plt.xlabel("Period")
plt.ylabel("Total Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

-------------------------------------------------------------------------------------------------------------------------------------------

forecast_final.select(
    [
        pl.col("effective_ratio").min().alias("min_ratio"),
        pl.col("effective_ratio").max().alias("max_ratio")
    ]
)

forecast_final.filter(
    pl.col("label").is_not_null()
).select(
    pl.col("label").unique()
)

(
    forecast_final
    .filter(
        (pl.col("periods") >= "2025 01") &
        (pl.col("periods") <= "2025 06")
    )
    .group_by("periods")
    .agg([
        (pl.col("final_forecast").sum() - pl.col("ma3").sum()).alias("diff")
    ])
)


(
    forecast_final
    .filter(
        (pl.col("periods") >= "2025 01") &
        (pl.col("periods") <= "2025 06")
    )
    .group_by("periods")
    .agg([
        (pl.col("final_forecast").sum() - pl.col("ma3").sum()).alias("diff")
    ])
)

forecast_final.select(
    pl.col("final_forecast").null_count()
)

forecast_final.filter(
    (pl.col("periods") >= "2025 01") &
    (pl.col("periods") <= "2025 03")
).select(
    pl.col("ma3").null_count()
)


(
    forecast_final
    .filter(
        (pl.col("periods") >= "2025 01") &
        (pl.col("periods") <= "2025 06")
    )
    .group_by("periods")
    .agg([
        pl.col("so_nw_ct").sum().alias("actual_sum"),
        pl.col("ma3").sum().alias("ma_sum"),
        pl.col("final_forecast").sum().alias("final_sum"),
        pl.count().alias("row_count")
    ])
    .sort("periods")
)

forecast_final.filter(
    pl.col("periods") == "2025 01"
).select([
    pl.col("so_nw_ct").sum(),
    pl.col("so_nw_ct").null_count(),
    pl.col("ma3").sum(),
    pl.col("final_forecast").sum()
])

forecast_final.filter(
    pl.col("periods") == "2025 01"
).select([
    pl.col("final_forecast").is_nan().sum().alias("nan_count"),
    pl.col("final_forecast").null_count().alias("null_count")
])

-------------------------------------------------------------------------------------------------------------------------------------------

# Cell 10: Compute MAPE for MA and Festive Adjusted

eval_df = (
    forecast_final
    .filter(pl.col("ma3").is_not_nan())  # drop invalid MA rows
    .with_columns([
        # Absolute error MA
        (pl.col("ma3") - pl.col("so_nw_ct")).abs().alias("ae_ma"),
        # Absolute error Festive
        (pl.col("final_forecast") - pl.col("so_nw_ct")).abs().alias("ae_festive"),
    ])
    .with_columns([
        # MAPE MA
        pl.when(pl.col("so_nw_ct") > 0)
        .then(pl.col("ae_ma") / pl.col("so_nw_ct"))
        .otherwise(None)
        .alias("mape_ma"),

        # MAPE Festive
        pl.when(pl.col("so_nw_ct") > 0)
        .then(pl.col("ae_festive") / pl.col("so_nw_ct"))
        .otherwise(None)
        .alias("mape_festive")
    ])
)

# Average per key
eval_key_df = (
    eval_df
    .group_by("key")
    .agg([
        pl.col("mape_ma").mean().alias("mape_ma_avg"),
        pl.col("mape_festive").mean().alias("mape_festive_avg")
    ])
)

eval_key_df.head()

-------------------------------------------------------------------------------------------------------------------------------------------

# Cell 11: Add Pareto flag

eval_key_df = eval_key_df.with_columns(
    pl.when(pl.col("key").is_in(pareto_df["key"]))
    .then(1)
    .otherwise(0)
    .alias("pareto80_flag")
)

eval_key_df.group_by("pareto80_flag").agg([
    pl.col("mape_ma_avg").mean().alias("ma_avg"),
    pl.col("mape_festive_avg").mean().alias("festive_avg")
])

-------------------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd

# Aggregate for plotting
plot_df = (
    eval_key_df
    .group_by("pareto80_flag")
    .agg([
        pl.col("mape_ma_avg").mean().alias("MA3"),
        pl.col("mape_festive_avg").mean().alias("Festive_Adjusted")
    ])
    .sort("pareto80_flag")
    .to_pandas()
)

# Plot
plot_df.set_index("pareto80_flag").plot(kind="bar", figsize=(8,5))

plt.title("MA3 vs Festive Adjusted MAPE\nPareto vs Non-Pareto")
plt.ylabel("Average MAPE")
plt.xticks([0,1], ["Non Pareto", "Pareto"], rotation=0)
plt.legend()
plt.show()

-------------------------------------------------------------------------------------------------------------------------------------------

# Filter 2023
df_2023 = df.filter(pl.col("periods").str.starts_with("2023"))

# Extract month number
df_2023 = df_2023.with_columns(
    pl.col("periods").str.split(" ").list.get(1).cast(pl.Int32).alias("month")
)

# Keep only Jan–Apr
df_2023 = df_2023.filter(pl.col("month").is_in([1,2,3,4]))


# Group by your key(s) and month
festive_avg_df = (
    df_2023.groupby(["key", "month"])
    .agg(pl.col("value").mean().alias("avg_value"))
)


# Ratio per month per key
# Assuming ratio = avg_value / sum(avg_value) per key
festive_ratio_df = (
    festive_avg_df.groupby("key")
    .agg([
        (pl.col("avg_value") / pl.col("avg_value").sum()).alias("ratio")
    ])
    .explode("ratio")  # optional if you need long format
)


# Reuse festive_ratio_long for labeling
# Example: month 4 is Lebaran
festive_ratio_long = (
    festive_ratio_df.with_columns(
        pl.when(pl.col("month") == 4).then("leb").otherwise("normal").alias("label")
    )
)


forecast_with_ratio = forecast.join(
    festive_ratio_long,
    on=["key", "month"],  # match by key and month
    how="left"
)
-------------------------------------------------------------------------------------------------------------------------------------------       


                            
-------------------------------------------------------------------------------------------------------------------------------------------



-------------------------------------------------------------------------------------------------------------------------------------------                            
