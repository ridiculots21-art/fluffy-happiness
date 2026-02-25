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

# Cell 6: combine MA forecast with per-key festive averages and plot sample key
# assumes: forecast_with_label, festive_avg_df, pl, pd, plt are already defined (from cells 1-5)

# 1) make festive_avg long (key, label, festive_avg) with labels matching lebaran_period_map ('3m','2m','1m','leb')
festive_long = (
    festive_avg_df
    .select([
        "key",
        "festive_avg_m3",
        "festive_avg_m2",
        "festive_avg_m1",
        "festive_avg_leb"
    ])
    # create columns with exact label names so melt produces '3m','2m','1m','leb'
    .with_columns([
        pl.col("festive_avg_m3").alias("3m"),
        pl.col("festive_avg_m2").alias("2m"),
        pl.col("festive_avg_m1").alias("1m"),
        pl.col("festive_avg_leb").alias("leb"),
    ])
    .select(["key", "3m", "2m", "1m", "leb"])
    .melt(id_vars="key", variable_name="label", value_name="festive_avg")
    .filter(~pl.col("festive_avg").is_null())  # keep only available festive averages
)

print("festive_long sample:")
display(festive_long.head(6).to_pandas())


# 2) join festival component onto forecast rows
# forecast_with_label should have: key, periods, ma3, so_nw_ct, lebaran_year (nullable), label (nullable)
forecast_with_festive = (
    forecast_with_label
    .join(festive_long, on=["key", "label"], how="left")
    # ensure periods parsed for plotting later
    .with_columns(
        pl.col("periods").str.strptime(pl.Date, "%Y %m").alias("period_dt")
    )
)

print("forecast_with_festive sample:")
display(forecast_with_festive.head(8).to_pandas())


# 3) create combined forecast: simple average of ma3 and festive_avg when festive_avg exists,
#    otherwise keep ma3 (you can change the combine rule later)
results_df = (
    forecast_with_festive
    .with_columns(
        combined_forecast = pl.when(pl.col("festive_avg").is_null())
            .then(pl.col("ma3"))
            .otherwise((pl.col("ma3") + pl.col("festive_avg")) / 2)
    )
    .select([
        "key", "periods", "period_dt", "label", "lebaran_year",
        "ma3", "festive_avg", "combined_forecast", "so_nw_ct"
    ])
    .sort(["key", "period_dt"])
)

print("results_df shape:", results_df.shape)
display(results_df.head(12).to_pandas())


# 4) quick plotting helper for a single key (change selected_key to any key you want to inspect)
selected_key = results_df["key"].unique()[0] if len(results_df) > 0 else None
print("selected_key (default):", selected_key)

if selected_key is not None:
    plot_df = results_df.filter(pl.col("key") == selected_key).to_pandas()
    # convert period_dt to pandas datetime if needed
    plot_df["period_dt"] = pd.to_datetime(plot_df["period_dt"])
    plot_df = plot_df.sort_values("period_dt")
    
    plt.figure(figsize=(10,4))
    plt.plot(plot_df["period_dt"], plot_df["so_nw_ct"], label="Actual so_nw_ct", marker="o")
    plt.plot(plot_df["period_dt"], plot_df["ma3"], label="MA3 forecast", marker="o")
    plt.plot(plot_df["period_dt"], plot_df["combined_forecast"], label="MA3 + FestiveAvg (mean)", marker="o", linestyle="--")
    plt.title(f"Key: {selected_key} — Actual vs MA3 vs Combined")
    plt.xlabel("Period")
    plt.ylabel("so_nw_ct / forecast")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No keys found in results_df to plot.")

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



-------------------------------------------------------------------------------------------------------------------------------------------


                            
-------------------------------------------------------------------------------------------------------------------------------------------       


                            
-------------------------------------------------------------------------------------------------------------------------------------------



-------------------------------------------------------------------------------------------------------------------------------------------                            
