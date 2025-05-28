import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("Grape Maturity Tracker")

st.sidebar.header("Harvest Readiness Rules")
brix_thresh = st.sidebar.number_input("Min Brix", value=24.0, step=0.1)
ta_thresh = st.sidebar.number_input("Max TA", value=6.0, step=0.1)
ph_min = st.sidebar.number_input("Min pH", value=3.0, step=0.01)
ph_max = st.sidebar.number_input("Max pH", value=4.0, step=0.01)

uploaded_files = st.file_uploader(
    "Upload Excel files", accept_multiple_files=True, type=['xlsx']
)

if uploaded_files:
    all_data = []
    for file in uploaded_files:
        year = file.name.split(" - ")[-1].replace(".xlsx", "")
        try:
            df_raw = pd.read_excel(file, sheet_name='Sheet1', header=7)
        except Exception as e:
            st.error(f"Error reading '{file.name}': {e}")
            continue
        df_raw.dropna(axis=1, how='all', inplace=True)
        df_raw.columns = df_raw.iloc[0].astype(str).str.strip()
        df = df_raw[1:].copy()
        df.columns = df.columns.str.strip()
        if df.columns.duplicated().any():
            st.error(
                f"Duplicate columns found in file {file.name}: "
                f"{df.columns[df.columns.duplicated()].tolist()}. "
                "Please fix the Excel file to remove duplicate column names."
            )
            continue
        df['Vintage'] = year
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        else:
            date_input = st.sidebar.date_input(f"Collection date for {year}")
            df['Date'] = pd.to_datetime(date_input)
        all_data.append(df)
    if not all_data:
        st.error("No valid data could be loaded from the uploaded files.")
    else:
        df_all = pd.concat(all_data, ignore_index=True)
        df_all.columns = df_all.columns.str.strip()
        # Ensure vineyard column exists for grouping
        if 'Vineyard' not in df_all.columns:
            st.error("No 'Vineyard' column found in the uploaded data. Please add a 'Vineyard' column.")
        else:
            for col in ['Brix', 'pH', 'TA', 'MA']:
                if col in df_all.columns:
                    df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

            st.sidebar.header("Filters")
            varieties = sorted(df_all['Variety'].dropna().astype(str).unique()) if 'Variety' in df_all.columns else []
            vineyards = sorted(df_all['Vineyard'].dropna().astype(str).unique())
            blocks = sorted(df_all['Block'].dropna().astype(str).unique()) if 'Block' in df_all.columns else []
            vintages = sorted(df_all['Vintage'].dropna().astype(str).unique())
            available_metrics = [col for col in ['Brix', 'pH', 'TA', 'MA'] if col in df_all.columns]
            vineyard = st.sidebar.selectbox("Vineyard", vineyards)
            block = st.sidebar.selectbox("Block", blocks)
            metric = st.sidebar.selectbox("Metric", available_metrics)
            vintage = st.sidebar.selectbox("Vintage (for Prediction)", vintages)

            # FILTER FOR SELECTED VINEYARD/BLOCK/METRIC
            filtered = df_all[
                (df_all['Vineyard'].astype(str) == str(vineyard)) &
                (df_all['Block'].astype(str) == str(block))
            ].copy()

            # --- 1. PREDICTION: Only use selected vintage ---
            st.subheader(f"Prediction for {vineyard} Block {block} ({vintage})")
            filtered_pred = filtered[filtered['Vintage'].astype(str) == str(vintage)].dropna(subset=[metric, 'Date']).copy()
            filtered_pred[metric] = pd.to_numeric(filtered_pred[metric], errors='coerce')
            filtered_pred = filtered_pred.dropna(subset=[metric])
            if len(filtered_pred) > 1:
                grp = filtered_pred.sort_values('Date')
                X = np.arange(len(grp)).reshape(-1, 1)
                y = grp[metric].values
                model = LinearRegression().fit(X, y)
                future = np.arange(len(grp), len(grp) + 5).reshape(-1, 1)
                preds = model.predict(future)
                dates = list(grp['Date']) + [grp['Date'].max() + pd.Timedelta(days=7 * (i + 1)) for i in range(5)]
                df_pred = pd.DataFrame({metric: np.concatenate([y, preds])}, index=dates)
                st.line_chart(df_pred, width=700, height=300)
                ready = np.array([])
                if metric == "Brix":
                    ready = np.where(preds >= brix_thresh)[0]
                elif metric == "TA":
                    ready = np.where(preds <= ta_thresh)[0]
                elif metric == "pH":
                    ready = np.where((preds >= ph_min) & (preds <= ph_max))[0]
                if ready.size > 0:
                    date_ready = dates[len(grp) + ready[0]]
                    st.success(f"Predicted readiness around {date_ready.date()}")
                else:
                    st.warning("No readiness predicted in forecast window.")
            else:
                st.warning("Not enough data points for prediction.")

            # --- 2. AGGREGATE PLOTS: Show all vintages for this block/vineyard ---
            st.subheader(f"All Vintages: {vineyard} Block {block} ({metric})")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9,5))
            for vtg in sorted(filtered['Vintage'].dropna().unique()):
                grp = filtered[filtered['Vintage'] == vtg].dropna(subset=['Date', metric])
                if len(grp):
                    ax.plot(grp['Date'], grp[metric], marker='o', label=str(vtg))
            ax.set_title(f"All Vintages for {vineyard} Block {block} ({metric})")
            ax.set_xlabel("Date")
            ax.set_ylabel(metric)
            ax.legend(title="Vintage", loc="best", fontsize='small')
            st.pyplot(fig)

            # --- 3. SUMMARY (optional, across all vintages) ---
            st.subheader("Summary (by Vineyard and Block)")
            summary_cols = [col for col in ['Brix', 'pH', 'TA', 'MA'] if col in filtered.columns]
            if summary_cols:
                summary = filtered.groupby(['Vineyard', 'Block'])[summary_cols].mean(numeric_only=True).round(2)
                st.dataframe(summary)
            else:
                st.info("No summary metrics available for the selected data.")
