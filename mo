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

# Keep only 2023 Jan–Apr
festive_df = so_fcst_df.filter(
    pl.col("periods").str.starts_with("2023")
).with_columns(
    month = pl.col("periods").str.split(" ").list.get(1).cast(pl.Int32)
).filter(pl.col("month").is_in([1,2,3,4]))

# Compute monthly average per key
festive_avg_df = festive_df.groupby(["key", "month"]).agg(
    pl.col("so_nw_ct").mean().alias("avg_value")
)

# Compute ratio per key
festive_ratio_df = festive_avg_df.groupby("key").agg(
    (pl.col("avg_value") / pl.col("avg_value").sum()).alias("ratio")
).explode("ratio")  # long format for easy merge

# Label months (example: month 4 = Lebaran, else normal)
festive_ratio_long = festive_avg_df.with_columns(
    label = pl.when(pl.col("month") == 4).then("leb").otherwise("normal")
)




# Take only 2023 Jan–Apr
festive_2023 = so_fcst_df.filter(
    (pl.col("periods").str.slice(0,4) == "2023") & 
    (pl.col("periods").str.slice(5,2).cast(pl.Int32) <= 4)
)

# Aggregate per key and month
festive_avg_df = festive_2023.groupby(["key","periods"]).agg(
    pl.col("so_nw_ct").mean().alias("avg_value")
)

# Extract month number
festive_avg_df = festive_avg_df.with_columns(
    month = pl.col("periods").str.slice(5,2).cast(pl.Int32)
)

# Map month → label
festive_ratio_long = festive_avg_df.with_columns(
    label = pl.when(pl.col("month") == 1).then(pl.lit("3m"))
            .when(pl.col("month") == 2).then(pl.lit("2m"))
            .when(pl.col("month") == 3).then(pl.lit("1m"))
            .when(pl.col("month") == 4).then(pl.lit("leb"))
)

# Compute ratio per key relative to month 4 (lebaran)
lebaran_values = festive_ratio_long.filter(pl.col("month") == 4).select(["key","avg_value"]).rename({"avg_value":"leb_value"})

festive_ratio_long = festive_ratio_long.join(lebaran_values, on="key", how="left")

festive_ratio_long = festive_ratio_long.with_columns(
    avg_value = (pl.col("avg_value") / pl.col("leb_value")).round(3)
).select(["key","label","avg_value"])

festive_ratio_long = festive_ratio_long.sort(["key","label"])

festive_ratio_long.head()




festive_ratio_long = festive_ratio_long.with_columns(
    festive_ratio = (pl.col("avg_value") / pl.col("leb_value"))
                     .fill_nan(1.0)      # replace NaN
                     .fill_null(1.0)     # replace missing
                     .clip_max(1e6)      # prevent Inf
                     .round(3)
).select(["key","label","festive_ratio"]).sort(["key","label"])
-------------------------------------------------------------------------------------------------------------------------------------------

normal_mean_df = (
    df_pareto
    .join(lebaran_period_map, on="periods", how="left")
    .filter(pl.col("label").is_null())   # non festive months only
    .group_by("key")
    .agg(
        pl.col("so_nw_ct").mean().alias("normal_mean")
    )
)


-------------------------------------------------------------------------------------------------------------------------------------------                            
# --- NEW CELL: Lebaran-only adjustment, plots, pareto bar (self-contained) ---
import numpy as np, polars as pl, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ---- 1) Prepare required sourceframes (try to reuse existing names) ----
# forecast_ma_df should have key, periods, ma3, so_nw_ct
if "forecast_ma_df" not in globals():
    # try to build minimal forecast_ma_df from train_df if available
    if "train_df" in globals():
        forecast_ma_df = (
            train_df.select(["key","periods","ma3","so_nw_ct"])
                    .sort(["key","periods"])
        )
    else:
        raise RuntimeError("forecast_ma_df not found and train_df not available to build it.")

# lebaran_period_map expected (periods <-> label mapping)
if "lebaran_period_map" not in globals():
    raise RuntimeError("lebaran_period_map not found. Please create it (map of lebaran_year,label,periods).")

# festive_ratio_long ideally exists (key,label,festive_ratio). If not, attempt to compute leb ratios only.
if "festive_ratio_long" in globals():
    fr_long = festive_ratio_long.select(["key","label","festive_ratio"])
else:
    # attempt to compute festive ratios minimal for leb using df_pareto or so_fcst_df + lebaran_period_map
    src_df = None
    if "df_pareto" in globals():
        src_df = df_pareto
    elif "so_fcst_df" in globals():
        src_df = so_fcst_df
    else:
        raise RuntimeError("Neither festive_ratio_long nor df_pareto/so_fcst_df found to compute festive ratios.")

    # compute festive averages per key+label using lebaran_period_map (restrict to label values present)
    tmp = (
        src_df.join(lebaran_period_map, on="periods", how="inner")  # only periods in map
              .group_by(["key","label"])
              .agg(pl.col("so_nw_ct").mean().alias("festive_avg"))
    )

    # compute a sensible 'normal' baseline per key (non-festive months within df used to compute normal mean)
    normal_mean = (
        src_df.join(lebaran_period_map, on="periods", how="left")
              .filter(pl.col("label").is_null())   # non-festive periods
              .group_by("key")
              .agg(pl.col("so_nw_ct").mean().alias("normal_mean"))
    )

    # join and compute ratio = festive_avg / normal_mean
    fr_long = (
        tmp.join(normal_mean, on="key", how="left")
           .with_columns(
               pl.when((pl.col("normal_mean").is_null()) | (pl.col("normal_mean") == 0))
                 .then(1.0)
                 .otherwise(pl.col("festive_avg") / pl.col("normal_mean"))
                 .alias("festive_ratio")
           )
           .select(["key","label","festive_ratio"])
    )

# Keep only the leb ratios (we only adjust leb months)
leb_ratio_df = fr_long.filter(pl.col("label") == "leb").select(["key","festive_ratio"]).rename({"festive_ratio":"leb_ratio"})

# Clean leb_ratio: replace non-finite or <=0 with 1.0
leb_ratio_df = leb_ratio_df.with_columns(
    leb_ratio = pl.when(~pl.col("leb_ratio").is_finite() | (pl.col("leb_ratio") <= 0))
                 .then(1.0)
                 .otherwise(pl.col("leb_ratio"))
)

# ---- 2) Annotate forecast rows with label (leb or not) ----
forecast_with_label_local = (
    forecast_ma_df.join(lebaran_period_map, on="periods", how="left")
                   .with_columns(pl.col("label").fill_null("").alias("label"))  # empty string when not festive
                   .sort(["key","periods"])
)

# join leb_ratio by key (leb_ratio will be null for keys that don't have leb_ratio)
forecast_with_label_local = forecast_with_label_local.join(leb_ratio_df, on="key", how="left")

# ---- 3) Build final forecast where ONLY leb periods are adjusted ----
# effective leb_ratio: use leb_ratio only when label == 'leb', otherwise 1.0
forecast_with_label_local = forecast_with_label_local.with_columns(
    leb_ratio_effective = pl.when(pl.col("label") == "leb")
                             .then(pl.col("leb_ratio"))
                             .otherwise(1.0)
)

# ensure no non-finite values remain
forecast_with_label_local = forecast_with_label_local.with_columns(
    leb_ratio_effective = pl.when(~pl.col("leb_ratio_effective").is_finite() | (pl.col("leb_ratio_effective") <= 0))
                             .then(1.0)
                             .otherwise(pl.col("leb_ratio_effective"))
)

# festive_adjusted_forecast (simple multiply) and smoothed final_forecast for leb only
forecast_with_label_local = forecast_with_label_local.with_columns([
    (pl.col("ma3") * pl.col("leb_ratio_effective")).alias("festive_adjusted_forecast"),
    # final smoothed forecast: if leb use smoothed formula else ma3 (we will create final_forecast accordingly)
    pl.when(pl.col("label") == "leb")
      .then(pl.col("ma3") * ((1 + pl.col("leb_ratio_effective")) / 2))
      .otherwise(pl.col("ma3"))
      .alias("final_forecast")
])

# ---- 4) Line chart (overall): Actual total, MA3 total, Final total where final uses adjusted only for leb months ----
agg_plot_df = (
    forecast_with_label_local
    .with_columns(period_dt = pl.col("periods").str.strptime(pl.Date, "%Y %m"))
    .group_by("period_dt")
    .agg([
        pl.col("so_nw_ct").sum().alias("actual_total"),
        pl.col("ma3").sum().alias("ma_total"),
        pl.col("final_forecast").sum().alias("final_total")
    ])
    .sort("period_dt")
    .to_pandas()
)

plt.figure(figsize=(11,5))
plt.plot(agg_plot_df["period_dt"], agg_plot_df["actual_total"], marker="o", label="Actual Total")
plt.plot(agg_plot_df["period_dt"], agg_plot_df["ma_total"], marker="o", label="MA3 Total")
plt.plot(agg_plot_df["period_dt"], agg_plot_df["final_total"], marker="o", linestyle="--", label="Festive-Adjusted (Leb only) Total")
plt.title("Overall Forecast Comparison — Final uses adjustment only for Leb")
plt.xlabel("Period")
plt.ylabel("Total Sales")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# ---- 5) Lebaran-only evaluation per key (MAPE) and Pareto bar chart ----
# select only rows where label == 'leb' (these are the leb months we adjusted)
leb_eval = forecast_with_label_local.filter(pl.col("label") == "leb")

# Compute AE and MAPE for MA and Final
leb_eval = leb_eval.with_columns([
    (pl.col("ma3") - pl.col("so_nw_ct")).abs().alias("ae_ma"),
    (pl.col("final_forecast") - pl.col("so_nw_ct")).abs().alias("ae_final")
]).with_columns([
    pl.when(pl.col("so_nw_ct") > 0).then(pl.col("ae_ma") / pl.col("so_nw_ct")).otherwise(None).alias("mape_ma"),
    pl.when(pl.col("so_nw_ct") > 0).then(pl.col("ae_final") / pl.col("so_nw_ct")).otherwise(None).alias("mape_final")
])

# average per key (Leb only)
eval_key_leban = (
    leb_eval.group_by("key")
            .agg([
                pl.col("mape_ma").mean().alias("mape_ma_avg"),
                pl.col("mape_final").mean().alias("mape_final_avg"),
                pl.count().alias("n_leb_rows")
            ])
)

# attach pareto flag using pareto_df if available (otherwise create a pareto flag False)
if "pareto_df" in globals():
    eval_key_leban = eval_key_leban.with_columns(
        pareto80_flag = pl.when(pl.col("key").is_in(pareto_df["key"])).then(1).otherwise(0)
    )
else:
    eval_key_leban = eval_key_leban.with_columns(pl.lit(0).alias("pareto80_flag"))

# Replace null mape means with large number or NaN? We'll leave null as-is so plotting ignores keys with no leb data
eval_key_leban = eval_key_leban.with_columns(
    mape_ma_avg = pl.col("mape_ma_avg").fill_null(pl.lit(np.nan)),
    mape_final_avg = pl.col("mape_final_avg").fill_null(pl.lit(np.nan))
)

# Aggregate for plotting (group by pareto flag)
plot_df = (
    eval_key_leban.group_by("pareto80_flag")
                  .agg([
                      pl.col("mape_ma_avg").mean().alias("MA3"),
                      pl.col("mape_final_avg").mean().alias("Festive_Adjusted")
                  ])
                  .sort("pareto80_flag")
                  .to_pandas()
)

# Bar plot: Leb-only Pareto vs Non-Pareto
plot_df = plot_df.set_index("pareto80_flag").rename(index={0:"Non Pareto",1:"Pareto"})
ax = plot_df.plot(kind="bar", figsize=(7,5))
ax.set_ylabel("Average Leb MAPE")
ax.set_title("Lebaran-only: MA3 vs Festive-Adjusted (averaged per key)\nPareto vs Non-Pareto")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.show()

# ---- 6) Return useful frames to the notebook namespace ----
# Keep results accessible for inspection
globals()["forecast_leban_final_df"] = forecast_with_label_local
globals()["leb_eval_per_key"] = eval_key_leban

print("Done — created `forecast_leban_final_df` (all rows with final_forecast) and `leb_eval_per_key` (Leb-only per-key MAPE).")




-----------------------------------------------------------

print("Keys in leb_eval:")
print(leb_eval.select(pl.col("key").n_unique()))

print("Keys in pareto_df:")
print(pareto_df.select(pl.col("key").n_unique()))

print("Overlap:")
print(
    leb_eval.select("key")
    .unique()
    .join(pareto_df.select("key").unique(), on="key", how="inner")
    .height
)


-----------------------------------------------------------

            plot_df = (
    forecast_final
    .with_columns(period_dt = pl.col("periods").str.strptime(pl.Date, "%Y %m"))
    .group_by("period_dt")
    .agg([
        pl.col("so_nw_ct").sum().alias("actual_total"),
        pl.col("ma3").sum().alias("ma_total"),
        pl.col("final_forecast").sum().alias("final_total")
    ])
    .sort("period_dt")
    .to_pandas()
)

plt.figure(figsize=(10,5))
plt.plot(plot_df["period_dt"], plot_df["actual_total"], marker="o", label="Actual")
plt.plot(plot_df["period_dt"], plot_df["ma_total"], marker="o", label="MA3")
plt.plot(plot_df["period_dt"], plot_df["final_total"], marker="o", linestyle="--", label="Festive Adjusted (Leb Only)")
plt.legend()
plt.title("Total Sales Comparison")
plt.grid(alpha=0.3)
plt.show()







leb_only = forecast_final.filter(pl.col("label") == "leb")

leb_only = leb_only.with_columns([
    (pl.col("ma3") - pl.col("so_nw_ct")).abs().alias("ae_ma"),
    (pl.col("final_forecast") - pl.col("so_nw_ct")).abs().alias("ae_final")
]).with_columns([
    (pl.col("ae_ma") / pl.col("so_nw_ct")).alias("mape_ma"),
    (pl.col("ae_final") / pl.col("so_nw_ct")).alias("mape_final")
])

per_key = leb_only.group_by("key").agg([
    pl.col("mape_ma").mean().alias("mape_ma_avg"),
    pl.col("mape_final").mean().alias("mape_final_avg")
])

per_key = per_key.with_columns(
    pareto_flag = pl.when(pl.col("key").is_in(pareto_df["key"]))
                    .then("Pareto")
                    .otherwise("Non Pareto")
)

plot_df = (
    per_key
    .group_by("pareto_flag")
    .agg([
        pl.col("mape_ma_avg").mean().alias("MA3"),
        pl.col("mape_final_avg").mean().alias("Festive_Adjusted")
    ])
    .to_pandas()
    .set_index("pareto_flag")
)

plot_df.plot(kind="bar", figsize=(6,4))
plt.title("Leb Month MAPE: Pareto vs Non-Pareto")
plt.ylabel("Average MAPE")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.show()

            
-----------------------------------------------------------            

# --- NEW CELL: Pareto bar (MA3 everywhere except leb -> final_forecast for leb) ---
import matplotlib.pyplot as plt
import pandas as pd

# evaluation window
periods_eval = [f"2025 {m:02d}" for m in range(1, 7)]

# sanity check: objects exist
if "forecast_final" not in globals():
    raise RuntimeError("forecast_final not found - run previous cells to create it first.")
if "pareto_df" not in globals():
    raise RuntimeError("pareto_df not found - run the pareto creation cell first.")

# select eval window and required cols
df_eval = (
    forecast_final
    .filter(pl.col("periods").is_in(periods_eval))
    .select([
        "key", "periods", "label", "ma3", "final_forecast", "so_nw_ct"
    ])
)

# choose forecast value:
# - if label == 'leb' and final_forecast exists -> use final_forecast
# - otherwise use ma3
df_eval = df_eval.with_columns(
    forecast_used = pl.when(
        (pl.col("label") == "leb") & pl.col("final_forecast").is_not_null()
    ).then(pl.col("final_forecast")).otherwise(pl.col("ma3"))
)

# compute absolute error and MAPE (only where actual > 0). cap MAPE at 1 (100%)
df_eval = df_eval.with_columns(
    ae = (pl.col("forecast_used") - pl.col("so_nw_ct")).abs(),
    mape = pl.when(pl.col("so_nw_ct") > 0)
            .then(pl.col("ae") / pl.col("so_nw_ct"))
            .otherwise(None)
).with_columns(
    mape = pl.when(pl.col("mape") > 1).then(1).otherwise(pl.col("mape"))
)

# drop keys that have no valid mape at all (optional, keeps averages meaningful)
per_key = (
    df_eval
    .group_by("key")
    .agg(
        pl.col("mape").mean().alias("mape_avg"),
        pl.col("mape").null_count().alias("mape_nulls"),
        pl.col("mape").count().alias("n_periods_evaluated")
    )
)

# attach pareto flag (1 = pareto, 0 = non-pareto)
per_key = per_key.with_columns(
    pareto80_flag = pl.when(per_key["key"].is_in(pareto_df["key"])).then(1).otherwise(0)
)

# keep only keys that had at least one non-null mape (n_periods_evaluated - mape_nulls > 0)
per_key = per_key.filter((pl.col("n_periods_evaluated") - pl.col("mape_nulls")) > 0)

# summary by pareto flag
summary = (
    per_key
    .group_by("pareto80_flag")
    .agg(
        pl.col("mape_avg").mean().alias("avg_mape"),
        pl.count().alias("comb_count")
    )
    .sort("pareto80_flag")
)

# convert to pandas and plot (showing percent)
pdf = summary.to_pandas().set_index("pareto80_flag")
pdf.index = pdf.index.map({0: "Non Pareto", 1: "Pareto"})
pdf["avg_mape_pct"] = pdf["avg_mape"] * 100

# basic bar plot
ax = pdf["avg_mape_pct"].plot(kind="bar", figsize=(6,4), rot=0)
ax.set_ylabel("Average MAPE (%)")
ax.set_title("Pareto vs Non-Pareto (MA3 except Leb -> final_forecast for Leb)")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom", fontsize=9)

plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()

# also show table for quick inspection
print("\nSummary table (avg_mape as fraction and key counts):")
display(summary)
            
-----------------------------------------------------------            

# ---------------------------
# CONFIG: choose which adjusted forecast to use for Lebaran month
# options: 'final' -> use forecast_final["final_forecast"]
#          'multiplicative' -> use forecast_final["festive_adjusted_forecast"]
LEBARAN_USE = 'final'   # set to 'multiplicative' if you prefer the raw multiplicative adjustment

# lebaran period and forecast horizon
LEBARAN_PERIOD = "2025 03"
HORIZON_START = "2025 01"
HORIZON_END   = "2025 06"

# pick column name for the lebaran adjusted forecast
if LEBARAN_USE == 'final':
    lebaran_col = "final_forecast"
else:
    lebaran_col = "festive_adjusted_forecast"

# ---------- construct the "pareto-mixed" forecast series ----------
# ensure forecast_final contains the columns we need: key, periods, ma3, {lebaran_col}, so_nw_ct
needed_cols = {"key","periods","ma3","so_nw_ct", lebaran_col}
missing = needed_cols - set(forecast_final.columns)
if missing:
    raise RuntimeError(f"Missing required columns in forecast_final: {missing}")

# restrict to forecast horizon (2025-01 .. 2025-06)
horizon_df = forecast_final.filter(
    (pl.col("periods") >= HORIZON_START) & (pl.col("periods") <= HORIZON_END)
).select(["key","periods","ma3","so_nw_ct", lebaran_col])

# build pareto_forecast: use lebaran_col only for the lebaran period, otherwise ma3
horizon_df = horizon_df.with_columns(
    pl.when((pl.col("periods") == LEBARAN_PERIOD) & pl.col(lebaran_col).is_not_null())
      .then(pl.col(lebaran_col))
      .otherwise(pl.col("ma3"))
      .alias("pareto_forecast")
)

# ---------- evaluate: MAPE for MA3 (baseline) and MAPE for Pareto-mixed ----------
eval_df = horizon_df.with_columns([
    # absolute errors
    (pl.col("ma3") - pl.col("so_nw_ct")).abs().alias("ae_ma"),
    (pl.col("pareto_forecast") - pl.col("so_nw_ct")).abs().alias("ae_pareto_mixed")
]).with_columns([
    # MAPE (cap at 1 and handle zero actuals)
    pl.when(pl.col("so_nw_ct") > 0)
      .then(pl.col("ae_ma") / pl.col("so_nw_ct"))
      .otherwise(None)
      .alias("mape_ma"),

    pl.when(pl.col("so_nw_ct") > 0)
      .then(pl.col("ae_pareto_mixed") / pl.col("so_nw_ct"))
      .otherwise(None)
      .alias("mape_pareto_mixed")
]).with_columns([
    pl.when(pl.col("mape_ma") > 1).then(1).otherwise(pl.col("mape_ma")).alias("mape_ma"),
    pl.when(pl.col("mape_pareto_mixed") > 1).then(1).otherwise(pl.col("mape_pareto_mixed")).alias("mape_pareto_mixed")
])

# ---------- aggregate per key ----------
eval_key_df = eval_df.group_by("key").agg([
    pl.col("mape_ma").mean().alias("mape_ma_avg"),
    pl.col("mape_pareto_mixed").mean().alias("mape_pareto_mixed_avg")
])

# flag Pareto80 based on your pareto_df (the same pareto_df you created earlier)
eval_key_df = eval_key_df.with_columns(
    pl.when(pl.col("key").is_in(pareto_df["key"])).then(1).otherwise(0).alias("pareto80_flag")
)

# ---------- aggregate for plotting: group-by pareto80_flag ----------
plot_df = (
    eval_key_df
    .group_by("pareto80_flag")
    .agg([
        pl.col("mape_ma_avg").mean().alias("MA3"),
        pl.col("mape_pareto_mixed_avg").mean().alias("ParetoMixed")
    ])
    .sort("pareto80_flag")
    .to_pandas()
)

# convert flag to label for nicer plotting
plot_df["group"] = plot_df["pareto80_flag"].map({0:"Non-Pareto", 1:"Pareto"})
plot_df = plot_df.set_index("group")[["MA3","ParetoMixed"]]

# ---------- print numeric comparison ----------
print("Average MAPE by group (Horizon: 2025-01 .. 2025-06). Lebaran month used from:", lebaran_col)
print(plot_df.round(4))

# ---------- plot bar chart ----------
import matplotlib.pyplot as plt
ax = plot_df.plot(kind="bar", figsize=(8,5))
ax.set_title("MA3 vs Pareto-mixed (Lebaran replaced by adjusted forecast)")
ax.set_ylabel("Average MAPE (capped at 1.0)")
ax.set_ylim(0, plot_df[["MA3","ParetoMixed"]].max().max()*1.15)
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()

-----------------------------------------------------------            

# --- New cell: Pareto graph using MA3 for all months except Lebaran (2025-03) -> use final_forecast ---
# (Comment / remove later if you just want to keep this as an experiment)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# config
LEBARAN_PERIOD = "2025 03"
HORIZON_START = "2025 01"
HORIZON_END   = "2025 06"

# ensure forecast_final has needed columns
_needed = {"key","periods","ma3","final_forecast","so_nw_ct"}
_missing = _needed - set(forecast_final.columns)
if _missing:
    raise RuntimeError(f"Missing columns in forecast_final: {_missing}")

# take only forecast horizon rows
horizon_df = forecast_final.filter(
    (pl.col("periods") >= HORIZON_START) & (pl.col("periods") <= HORIZON_END)
).select(["key","periods","ma3","final_forecast","so_nw_ct"])

# build the pareto-mixed forecast: use final_forecast only for Lebaran period, otherwise ma3
horizon_df = horizon_df.with_columns(
    pl.when(pl.col("periods") == LEBARAN_PERIOD)
      .then(pl.col("final_forecast"))
      .otherwise(pl.col("ma3"))
      .alias("pareto_mixed_forecast")
)

# compute AE and MAPE (same logic as your code, cap at 1)
eval_df = horizon_df.with_columns([
    (pl.col("ma3") - pl.col("so_nw_ct")).abs().alias("ae_ma"),
    (pl.col("pareto_mixed_forecast") - pl.col("so_nw_ct")).abs().alias("ae_pareto")
]).with_columns([
    pl.when(pl.col("so_nw_ct") > 0).then(pl.col("ae_ma") / pl.col("so_nw_ct")).otherwise(None).alias("mape_ma"),
    pl.when(pl.col("so_nw_ct") > 0).then(pl.col("ae_pareto") / pl.col("so_nw_ct")).otherwise(None).alias("mape_pareto")
]).with_columns([
    pl.when(pl.col("mape_ma") > 1).then(1).otherwise(pl.col("mape_ma")).alias("mape_ma"),
    pl.when(pl.col("mape_pareto") > 1).then(1).otherwise(pl.col("mape_pareto")).alias("mape_pareto")
])

# per-key average MAPE across the horizon
eval_key_df = eval_df.group_by("key").agg([
    pl.col("mape_ma").mean().alias("mape_ma_avg"),
    pl.col("mape_pareto").mean().alias("mape_pareto_avg")
])

# mark Pareto80 using your existing pareto_df
eval_key_df = eval_key_df.with_columns(
    pl.when(pl.col("key").is_in(pareto_df["key"])).then(1).otherwise(0).alias("pareto80_flag")
)

# aggregate for plot (mean MAPE per group)
plot_df = (
    eval_key_df
    .group_by("pareto80_flag")
    .agg([
        pl.col("mape_ma_avg").mean().alias("MA3"),
        pl.col("mape_pareto_avg").mean().alias("ParetoMixed")
    ])
    .sort("pareto80_flag")
    .to_pandas()
)

# label groups
plot_df["group"] = plot_df["pareto80_flag"].map({0:"Non-Pareto", 1:"Pareto"})
plot_df = plot_df.set_index("group")[["MA3","ParetoMixed"]]

# show numeric table
print("Average MAPE by group (Horizon: 2025-01 .. 2025-06). Lebaran (2025-03) uses final_forecast:")
display(plot_df.round(4))

# plot bar chart
ax = plot_df.plot(kind="bar", figsize=(8,5))
ax.set_title("MA3 vs Pareto-mixed (Lebaran replaced by final_forecast)")
ax.set_ylabel("Average MAPE (capped at 1.0)")
ax.set_ylim(0, max(plot_df.max().max()*1.15, 1.0))
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()

# (Optional) export eval_key_df if you want to inspect per-key changes
# eval_key_df.write_csv("eval_key_pareto_mixed_vs_ma3.csv")
















# Pareto-specific evaluation: use final_forecast (MA3 except Lebaran already replaced)

pareto_eval_df = (
    forecast_final
    .filter(pl.col("final_forecast").is_not_nan())
    .with_columns([
        (pl.col("final_forecast") - pl.col("so_nw_ct")).abs().alias("ae_pareto")
    ])
    .with_columns([
        pl.when(pl.col("so_nw_ct") > 0)
        .then(pl.col("ae_pareto") / pl.col("so_nw_ct"))
        .otherwise(None)
        .alias("mape_pareto")
    ])
)

pareto_eval_key_df = (
    pareto_eval_df
    .group_by("key")
    .agg(
        pl.col("mape_pareto").mean().alias("mape_pareto_avg")
    )
)

pareto_eval_key_df = pareto_eval_key_df.with_columns(
    pl.when(pl.col("key").is_in(pareto_df["key"]))
    .then(1)
    .otherwise(0)
    .alias("pareto80_flag")
)

pareto_eval_key_df.group_by("pareto80_flag").agg(
    pl.col("mape_pareto_avg").mean().alias("pareto_mape_avg")
)           































# Adjusted Pareto evaluation:
# Use MA3 normally, but replace rows where label contains "leb" with final_forecast

adjusted_eval_df = (
    forecast_final
    .filter(pl.col("ma3").is_not_nan())
    .with_columns([
        pl.when(pl.col("label").str.contains("leb"))
        .then(pl.col("final_forecast"))
        .otherwise(pl.col("ma3"))
        .alias("adjusted_forecast")
    ])
    .with_columns([
        (pl.col("adjusted_forecast") - pl.col("so_nw_ct")).abs().alias("ae_adjusted")
    ])
    .with_columns([
        pl.when(pl.col("so_nw_ct") > 0)
        .then(pl.col("ae_adjusted") / pl.col("so_nw_ct"))
        .otherwise(None)
        .alias("mape_adjusted")
    ])
)

adjusted_eval_key_df = (
    adjusted_eval_df
    .group_by("key")
    .agg(
        pl.col("mape_adjusted").mean().alias("mape_adjusted_avg")
    )
)

adjusted_eval_key_df = adjusted_eval_key_df.with_columns(
    pl.when(pl.col("key").is_in(pareto_df["key"]))
    .then(1)
    .otherwise(0)
    .alias("pareto80_flag")
)

adjusted_eval_key_df.group_by("pareto80_flag").agg(
    pl.col("mape_adjusted_avg").mean().alias("adjusted_mape_avg")
)








            
# Adjusted Pareto evaluation:
# Use MA3 normally, but replace rows where label contains "leb" with final_forecast

adjusted_eval_df = (
    forecast_final
    .filter(pl.col("ma3").is_not_nan())
    .with_columns([
        pl.when(pl.col("label").str.contains("leb"))
        .then(pl.col("final_forecast"))
        .otherwise(pl.col("ma3"))
        .alias("adjusted_forecast")
    ])
    .with_columns([
        (pl.col("adjusted_forecast") - pl.col("so_nw_ct")).abs().alias("ae_adjusted")
    ])
    .with_columns([
        pl.when(pl.col("so_nw_ct") > 0)
        .then(pl.col("ae_adjusted") / pl.col("so_nw_ct"))
        .otherwise(None)
        .alias("mape_adjusted")
    ])
)

adjusted_eval_key_df = (
    adjusted_eval_df
    .group_by("key")
    .agg(
        pl.col("mape_adjusted").mean().alias("mape_adjusted_avg")
    )
)

adjusted_eval_key_df = adjusted_eval_key_df.with_columns(
    pl.when(pl.col("key").is_in(pareto_df["key"]))
    .then(1)
    .otherwise(0)
    .alias("pareto80_flag")
)

adjusted_eval_key_df.group_by("pareto80_flag").agg(
    pl.col("mape_adjusted_avg").mean().alias("adjusted_mape_avg")
)







adjusted_eval_df = (
    forecast_final
    .filter(
        pl.when(pl.col("label") == "leb")
        .then(pl.col("final_forecast").is_not_nan())
        .otherwise(pl.col("ma3").is_not_nan())
    )
    .with_columns([
        pl.when(pl.col("label") == "leb")
        .then(pl.col("final_forecast"))
        .otherwise(pl.col("ma3"))
        .alias("adjusted_forecast")
    ])
    .with_columns([
        (pl.col("adjusted_forecast") - pl.col("so_nw_ct")).abs().alias("ae_adjusted")
    ])
    .with_columns([
        pl.when(pl.col("so_nw_ct") > 0)
        .then(pl.col("ae_adjusted") / pl.col("so_nw_ct"))
        .otherwise(None)
        .alias("mape_adjusted")
    ])
)

adjusted_eval_key_df = (
    adjusted_eval_df
    .group_by("key")
    .agg(
        pl.col("mape_adjusted").mean().alias("mape_adjusted_avg")
    )
)

adjusted_eval_key_df = adjusted_eval_key_df.with_columns(
    pl.when(pl.col("key").is_in(pareto_df["key"]))
    .then(1)
    .otherwise(0)
    .alias("pareto80_flag")
)

adjusted_eval_key_df.group_by("pareto80_flag").agg(
    pl.col("mape_adjusted_avg").mean().alias("adjusted_mape_avg")
)







       



adjusted_eval_df = (
    forecast_final
    .filter(
        pl.when(pl.col("label") == "leb")
        .then(pl.col("final_forecast").is_not_nan())
        .otherwise(pl.col("ma3").is_not_nan())
    )
    .with_columns([
        pl.when(pl.col("label") == "leb")
        .then(pl.col("final_forecast"))
        .otherwise(pl.col("ma3"))
        .alias("adjusted_forecast")
    ])
    .with_columns([
        (pl.col("adjusted_forecast") - pl.col("so_nw_ct")).abs().alias("ae_adjusted"),
    ])
    .with_columns([
        (pl.col("ae_adjusted") / pl.col("so_nw_ct"))
        .alias("mape_adjusted")
    ])
)










adjusted_eval_df = (
    forecast_final
    .with_columns([
        pl.when(pl.col("label") == "leb")
        .then(pl.col("final_forecast"))
        .otherwise(pl.col("ma3"))
        .alias("adjusted_forecast")
    ])
    .filter(pl.col("adjusted_forecast").is_not_nan())
    .with_columns([
        (pl.col("adjusted_forecast") - pl.col("so_nw_ct")).abs().alias("ae_adjusted")
    ])
    .with_columns([
        pl.when(pl.col("so_nw_ct") > 0)
        .then(pl.col("ae_adjusted") / pl.col("so_nw_ct"))
        .otherwise(None)
        .alias("mape_adjusted")
    ])
)

adjusted_eval_key_df = (
    adjusted_eval_df
    .group_by("key")
    .agg(
        pl.col("mape_adjusted").mean().alias("mape_adjusted_avg")
    )
)

adjusted_eval_key_df = adjusted_eval_key_df.with_columns(
    pl.when(pl.col("key").is_in(pareto_df["key"]))
    .then(1)
    .otherwise(0)
    .alias("pareto80_flag")
)






forecast_final.filter(
    pl.col("label") == "leb"
).select(
    pl.col("key")
).unique().join(
    pareto_df.select("key").unique(),
    on="key",
    how="inner"
).shape






print(
    "Total leb keys:",
    forecast_final.filter(pl.col("label") == "leb").select("key").n_unique()
)

print(
    "Leb keys in Pareto:",
    forecast_final.filter(pl.col("label") == "leb")
    .join(pareto_df.select("key").unique(), on="key", how="inner")
    .select("key")
    .n_unique()
)            

adjusted_eval_key_df.group_by("pareto80_flag").agg(
    pl.col("mape_adjusted_avg").mean().alias("adjusted_mape_avg")
)            




comparison_df = (
    pareto_eval_key_df
    .join(adjusted_eval_key_df, on=["key", "pareto80_flag"])
    .with_columns(
        (pl.col("mape_adjusted_avg") - pl.col("mape_pareto_avg")).alias("diff")
    )
)

comparison_df.select([
    pl.col("diff").abs().max().alias("max_diff"),
    pl.col("diff").abs().mean().alias("mean_diff"),
    pl.col("diff").abs().sum().alias("total_diff")
])




ma3_group = pareto_eval_key_df.group_by("pareto80_flag").agg(
    pl.col("mape_pareto_avg").mean().alias("ma3_avg")
)

adjusted_group = adjusted_eval_key_df.group_by("pareto80_flag").agg(
    pl.col("mape_adjusted_avg").mean().alias("adjusted_avg")
)

ma3_group.join(adjusted_group, on="pareto80_flag")            








comparison_group = (
    eval_key_df
    .select(["key", "pareto80_flag", "mape_festive_avg"])
    .join(
        adjusted_eval_key_df.select(["key", "pareto80_flag", "mape_adjusted_avg"]),
        on=["key", "pareto80_flag"],
        how="inner"
    )
)

# Compare group averages side-by-side
comparison_group.group_by("pareto80_flag").agg([
    pl.col("mape_festive_avg").mean().alias("festive_mape_avg"),
    pl.col("mape_adjusted_avg").mean().alias("adjusted_mape_avg"),
    (pl.col("mape_adjusted_avg") - pl.col("mape_festive_avg")).mean().alias("avg_diff")
])





comparison_group.select([
    (pl.col("mape_adjusted_avg") == pl.col("mape_festive_avg"))
    .all()
    .alias("all_keys_identical"),

    (pl.col("mape_adjusted_avg") - pl.col("mape_festive_avg"))
    .abs()
    .max()
    .alias("max_difference_per_key")
])            
