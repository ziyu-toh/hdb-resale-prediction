import polars as pl
import os

cwd = os.getcwd()
print("CWD: ", cwd)
DATA_PATH = os.path.join(cwd, "data/raw")
print("Data path: ", DATA_PATH)

# Load data
file_names = [x for x in os.listdir(DATA_PATH) if ".csv" in x]  # List all CSV files in the raw data directory

df_list = []
for file in file_names:
    print("Loading file:", file)
    df = pl.scan_csv(os.path.join(DATA_PATH, file), infer_schema=False)
    df_list.append(df)
print("Loaded", len(df_list), "files.")

# Combine lazyframes into a single lazyframe
df_combined = pl.concat(df_list, how="align_full")
df_combined_col = df_combined.collect()

# Convert variables to appropriate types
df_combined_col = df_combined_col.with_columns(
    pl.col("resale_price").cast(pl.Float64),
    pl.col("lease_commence_date").cast(pl.Int64),
    pl.col("floor_area_sqm").cast(pl.Float64),
    pl.col("month").str.to_date("%Y-%m")
)

# Calculate months from earliest month in dataset (Mar-2012)
df_combined_col = df_combined_col.with_columns(
    flat_age_years = (pl.col("month").dt.year() - pl.col("lease_commence_date")),
    days_from_earliest_data = (pl.col('month') - pl.date(2012, 3, 1)).dt.total_days()
).drop("remaining_lease") # can be dropped

# Cleaning flat model: Combine Maisonettes
df_combined_col = df_combined_col.with_columns(flat_model_revised = pl.when(pl.col("flat_model").str.contains("Maisonette"))
    .then(pl.lit("Maisonette"))
    .otherwise(pl.col("flat_model"))
)
df_combined_col.select(pl.col("flat_model_revised").unique()).sort("flat_model_revised")

# Create min max value for stories, then group every 15 storeys since a part of the data groups every 5 floors instead of 3 floors
df_combined_col = df_combined_col.with_columns(
    storey_max = pl.col("storey_range").str.slice(-2,2).cast(pl.Int64)
)

df_combined_col = df_combined_col.with_columns(
    storey_range_grouped = (pl.when(pl.col("storey_max")<= 15).then(pl.lit("0-15"))
                            .when(pl.col("storey_max").is_between(16, 30)).then(pl.lit("16-30"))
                            .otherwise(pl.lit("31+")))
)
df_combined_col.group_by("storey_range_grouped").agg(
    pl.col("storey_max").min().alias("min_storey"),
    pl.col("storey_max").max().alias("max_storey"),
    pl.col("storey_max").count().alias("counts")
).sort("storey_range_grouped")


# TODO: Other factors to KIV: Distance from MRT/amenities, lease years left, supply and demand of surrounding areas


# Keep relevant columns
df_output = df_combined_col.select(
        'month',
        'town',
        'flat_type',
        'flat_model_revised',
        'flat_age_years',
        'floor_area_sqm',
        'days_from_earliest_data',
        'storey_range_grouped',
        'resale_price',
)

# Split into train (2012-2023), test(2024) and deploy sets (2025)
train = df_output.filter(df_output['month'].dt.year().is_between(2012, 2023)).drop("month").write_parquet("data/processed/train.parquet")
test = df_output.filter(df_output['month'].dt.year() == 2024).drop("month").write_parquet("data/processed/test.parquet")
deploy = df_output.filter(df_output['month'].dt.year() > 2024).drop("month").write_parquet("data/processed/deploy.parquet")

print("Data processing complete.")