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

-------------------------------------------------------------------------------------------------------------------------------------------

# Cell 5 — Clean festive data preparation (NO combining)

festive_clean = (
    festive_raw
    .filter(pl.col("n_years_key").is_not_null())  # remove NaN years
    .with_columns([
        pl.col("lebaran_year").cast(pl.Int64)     # remove decimal .0
    ])
)

festive_clean

# CELL 5: attach per-key festive averages to forecast rows

forecast_enriched = (
    forecast_with_label
    .join(
        per_key_lebaran_avg,
        on="key",
        how="left"
    )
    .filter(pl.col("n_years_key").is_not_null())
)

print("Forecast enriched sample:")
display(forecast_enriched.head(10).to_pandas())

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



-------------------------------------------------------------------------------------------------------------------------------------------                            
