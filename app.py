import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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
    for i, file in enumerate(uploaded_files):
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
        # Ensure every Block is tied to the Vineyard in the cell to the left
        if 'Vineyard' in df.columns and 'Block' in df.columns:
            df['Vineyard'] = df['Vineyard'].ffill()
        # Check for duplicate columns and skip files that have them
        if df.columns.duplicated().any():
            st.error(
                f"Duplicate columns found in file {file.name}: "
                f"{df.columns[df.columns.duplicated()].tolist()}. "
                "Please fix the Excel file to remove duplicate column names."
            )
            continue
        df['Vintage'] = year
        # Date handling with robust parsing
        if 'Date' in df.columns:
            # Try parsing with explicit format for MM/DD/YY (e.g. 8/30/24)
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y', errors='coerce')
            # Fallback: try general parsing for any not caught above
            df['Date'] = df['Date'].fillna(pd.to_datetime(df['Date'], errors='coerce'))
            # Debug output (remove or comment out if not needed)
            st.write(f"Parsed dates for {file.name}:", df['Date'].head())
            st.write(f"Unique years in Date column for {file.name}:", df['Date'].dt.year.unique())
        else:
            st.warning(
                f"File '{file.name}' is missing a 'Date' column. "
                f"Please enter the correct collection date for this file below. "
                f"Incorrect dates may result in prediction errors.",
                icon="⚠️"
            )
            date_input = st.sidebar.date_input(
                f"Collection date for {year} ({file.name}) [REQUIRED]",
                value=None,
                key=f"date_input_{i}_{file.name}"
            )
            if date_input is None:
                st.stop()  # Force user to pick a date
            df['Date'] = pd.to_datetime(date_input)
        all_data.append(df)
    if not all_data:
        st.error("No valid data could be loaded from the uploaded files.")
    else:
        df_all = pd.concat(all_data, ignore_index=True)
        df_all.columns = df_all.columns.str.strip()
        # Ensure metrics are numeric
        for col in ['Brix', 'pH', 'TA', 'MA']:
            if col in df_all.columns:
                df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

        # --- FILTERING & FLIPPING FUNCTIONALITY ---
        st.sidebar.header("Vineyard and Block Navigation")
        if 'Vineyard' not in df_all.columns or 'Block' not in df_all.columns:
            st.error("Missing required columns: 'Vineyard' and/or 'Block'. Check your Excel file headers.")
            st.write("Columns found:", df_all.columns.tolist())
            st.stop()
        vineyards = sorted(df_all['Vineyard'].dropna().astype(str).unique())
        vineyard = st.sidebar.selectbox("Vineyard", vineyards, key="vineyard_select")
        blocks = sorted(df_all[df_all['Vineyard'].astype(str) == vineyard]['Block'].dropna().astype(str).unique())
        block = st.sidebar.selectbox("Block", blocks, key="block_select")
        # Flip-through feature: Prev/Next buttons for Vineyard/Block
        v_idx = vineyards.index(vineyard)
        b_idx = blocks.index(block)
        col1, col2, col3, col4 = st.sidebar.columns([1,1,1,1])
        with col1:
            if st.button("Prev Vineyard") and v_idx > 0:
                st.session_state['vineyard_select'] = vineyards[v_idx - 1]
        with col2:
            if st.button("Next Vineyard") and v_idx < len(vineyards)-1:
                st.session_state['vineyard_select'] = vineyards[v_idx + 1]
        with col3:
            if st.button("Prev Block") and b_idx > 0:
                st.session_state['block_select'] = blocks[b_idx - 1]
        with col4:
            if st.button("Next Block") and b_idx < len(blocks)-1:
                st.session_state['block_select'] = blocks[b_idx + 1]

        available_metrics = [col for col in ['Brix', 'pH', 'TA', 'MA'] if col in df_all.columns]
        metric = st.sidebar.selectbox("Metric", available_metrics)

        # --- FILTERED DATA ---
        filtered = df_all[
            (df_all['Vineyard'].astype(str) == vineyard) &
            (df_all['Block'].astype(str) == block)
        ].copy()

        # ---- 1. PREDICTION: Only current calendar year ----
        st.subheader(f"Prediction for {vineyard} Block {block} in {datetime.datetime.now().year}")
        current_year = datetime.datetime.now().year
        filtered_pred = filtered[
            filtered['Date'].dt.year == current_year
        ].dropna(subset=[metric, 'Date']).copy()
        filtered_pred[metric] = pd.to_numeric(filtered_pred[metric], errors='coerce')
        filtered_pred = filtered_pred.dropna(subset=[metric])
        if filtered_pred.empty:
            st.warning(
                f"No data for {current_year} in your uploaded files. "
                f"Please upload files with data from this year or enter correct dates.",
                icon="⚠️"
            )
        elif len(filtered_pred) > 1:
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
            st.warning("Not enough data points for prediction for the current year.")

        # ---- 2. AGGREGATE PLOTS: All vintages for this block/vineyard ----
        st.subheader(f"All Vintages: {vineyard} Block {block} ({metric})")
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

        # ---- 3. SUMMARY (by Vineyard and Block, all years) ----
        st.subheader("Summary (by Vineyard and Block, all vintages)")
        summary_cols = [col for col in ['Brix', 'pH', 'TA', 'MA'] if col in filtered.columns]
        if summary_cols:
            summary = filtered.groupby(['Vineyard', 'Block'])[summary_cols].mean(numeric_only=True).round(2)
            st.dataframe(summary)
        else:
            st.info("No summary metrics available for the selected data.")
