import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("Grape Maturity Tracker")

# Sidebar for harvest readiness rules
st.sidebar.header("Harvest Readiness Rules")
brix_thresh = st.sidebar.number_input("Min Brix", value=24.0, step=0.1)
ta_thresh = st.sidebar.number_input("Max TA", value=6.0, step=0.1)
ph_min = st.sidebar.number_input("Min pH", value=3.0, step=0.01)
ph_max = st.sidebar.number_input("Max pH", value=4.0, step=0.01)

# File upload
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
        # Set column names from first row and strip whitespace
        df_raw.columns = df_raw.iloc[0].astype(str).str.strip()
        df = df_raw[1:].copy()
        df.columns = df.columns.str.strip()
        df['Vintage'] = year
        # Date handling
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
        df_all.columns = df_all.columns.str.strip()  # Ensure columns are stripped after concat

        # Convert metrics if columns exist (do this after concat in case any non-numeric slipped in)
        for col in ['Brix', 'pH', 'TA', 'MA']:
            if col in df_all.columns:
                df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

        # Debug: Show all columns after concat
        st.write("All columns in concatenated data:", df_all.columns.tolist())

        # Filters
        st.sidebar.header("Filters")
        if 'Vintage' in df_all.columns and not df_all['Vintage'].dropna().empty:
            vintages = sorted(df_all['Vintage'].dropna().unique())
            selected_vintages = st.sidebar.multiselect("Vintages", vintages, default=vintages)
        else:
            st.error("No 'Vintage' column found or empty in the uploaded data.")
            selected_vintages = []

        # Defensive checks to avoid errors if columns are missing
        variety = None
        block = None
        metric = None

        if 'Variety' in df_all.columns:
            if not df_all['Variety'].dropna().empty:
                varieties = sorted(df_all['Variety'].dropna().astype(str).unique())
                variety = st.sidebar.selectbox("Variety", varieties)
            else:
                st.error("'Variety' column is present but has no data.")
        else:
            st.error("No 'Variety' column found in the uploaded data.")

        if 'Block' in df_all.columns:
            if not df_all['Block'].dropna().empty:
                blocks = sorted(df_all['Block'].dropna().astype(str).unique())
                block = st.sidebar.selectbox("Block", blocks)
            else:
                st.error("'Block' column is present but has no data.")
        else:
            st.error("No 'Block' column found in the uploaded data.")

        available_metrics = [col for col in ['Brix', 'pH', 'TA', 'MA'] if col in df_all.columns]
        if available_metrics:
            metric = st.sidebar.selectbox("Metric", available_metrics)
        else:
            st.error("None of the metric columns (Brix, pH, TA, MA) found in the uploaded data.")

        # Only continue if all necessary columns are found and have values
        if (
            variety and block and metric and
            selected_vintages and
            'Date' in df_all.columns
        ):
            filtered = df_all[
                (df_all['Vintage'].isin(selected_vintages)) &
                (df_all['Variety'].astype(str) == str(variety)) &
                (df_all['Block'].astype(str) == str(block))
            ].sort_values('Date')

            # Plot comparison
            st.subheader(f"{metric} Comparison")
            for vintage in selected_vintages:
                grp = filtered[filtered['Vintage'] == vintage]
                if not grp.empty:
                    # Defensive: Only plot if metric is available in group
                    if metric in grp.columns:
                        st.line_chart(grp.set_index('Date')[metric], width=700, height=300)
                    else:
                        st.warning(f"Metric '{metric}' not found in data for vintage {vintage}.")

            # Prediction
            st.subheader("Readiness Prediction")
            grp = filtered.dropna(subset=[metric, 'Date']).copy()
            grp[metric] = pd.to_numeric(grp[metric], errors='coerce')  # Ensure numeric
            grp = grp.dropna(subset=[metric])
            if len(grp) > 1:
                X = np.arange(len(grp)).reshape(-1, 1)
                y = grp[metric].values
                model = LinearRegression().fit(X, y)
                future = np.arange(len(grp), len(grp) + 5).reshape(-1, 1)
                preds = model.predict(future)
                # Extend dates list for predictions
                dates = list(grp['Date']) + [grp['Date'].max() + pd.Timedelta(days=7 * (i + 1)) for i in range(5)]
                df_pred = pd.DataFrame({metric: np.concatenate([y, preds])}, index=dates)
                st.line_chart(df_pred, width=700, height=300)
                # Check readiness
                ready = np.array([])
                if metric == "Brix":
                    ready = np.where(preds >= brix_thresh)[0]
                elif metric == "TA":
                    ready = np.where(preds <= ta_thresh)[0]
                elif metric == "pH":
                    ready = np.where((preds >= ph_min) & (preds <= ph_max))[0]
                # For MA or other metrics, no readiness
                if ready.size > 0:
                    date_ready = dates[len(grp) + ready[0]]
                    st.success(f"Predicted readiness around {date_ready.date()}")
                else:
                    st.warning("No readiness predicted in forecast window.")
            else:
                st.warning("Not enough data points for prediction.")

            # Summary
            st.subheader("Summary")
            summary_cols = [col for col in ['Brix', 'pH', 'TA', 'MA'] if col in filtered.columns]
            if summary_cols:
                summary = filtered.groupby('Vintage')[summary_cols].mean(numeric_only=True).round(2)
                st.dataframe(summary)
            else:
                st.info("No summary metrics available for the selected data.")
