import sqlite3
import pandas as pd
import os
DB_PATH = "./infra/mlflow_data/mlflow.db"

if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"Not found database in: {DB_PATH}")

conn = sqlite3.connect(DB_PATH)

def export_table(table_name):
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        csv_file = f"{table_name}.csv"
        df.to_csv(csv_file, index=False)
        print(f" Export table '{table_name}' to {csv_file} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"No export '{table_name}': {e}")
        return pd.DataFrame()

runs_df     = export_table("runs")
metrics_df  = export_table("metrics")
params_df   = export_table("params")
tags_df     = export_table("tags")
latest_df   = export_table("latest_metrics")

if not runs_df.empty and not metrics_df.empty and not params_df.empty:
    summary = runs_df[['run_uuid', 'experiment_id', 'status', 'start_time', 'end_time']]

    metrics_pivot = metrics_df.pivot_table(index='run_uuid', 
                                           columns='key', 
                                           values='value', 
                                           aggfunc='first').reset_index()

    params_pivot = params_df.pivot_table(index='run_uuid', 
                                         columns='key', 
                                         values='value', 
                                         aggfunc='first').reset_index()

    summary = summary.merge(metrics_pivot, on='run_uuid', how='left')
    summary = summary.merge(params_pivot, on='run_uuid', how='left')

    summary.to_csv("summary.csv", index=False)
    print(f"Summary: summary.csv ({len(summary)} rows)")
else:
    print(" Can't generated data in 'runs', 'metrics' or 'params'.")

conn.close()
