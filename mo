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

# Cell 9: build final averaged forecast

forecast_final = (
    forecast_adjusted
    .with_columns(
        (
            (pl.col("ma3") + pl.col("festive_adjusted_forecast")) / 2
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
    "festive_adjusted_forecast",
    "final_forecast",
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



-------------------------------------------------------------------------------------------------------------------------------------------



-------------------------------------------------------------------------------------------------------------------------------------------



-------------------------------------------------------------------------------------------------------------------------------------------



-------------------------------------------------------------------------------------------------------------------------------------------


-------------------------------------------------------------------------------------------------------------------------------------------



-------------------------------------------------------------------------------------------------------------------------------------------


                            
-------------------------------------------------------------------------------------------------------------------------------------------       


                            
-------------------------------------------------------------------------------------------------------------------------------------------



-------------------------------------------------------------------------------------------------------------------------------------------                            
