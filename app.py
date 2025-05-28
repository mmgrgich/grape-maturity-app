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
uploaded_files = st.file_uploader("Upload Excel files", accept_multiple_files=True, type=['xlsx'])

if uploaded_files:
    all_data = []
    for file in uploaded_files:
        year = file.name.split(" - ")[-1].replace(".xlsx", "")
        df_raw = pd.read_excel(file, sheet_name='Sheet1', header=7)
        df_raw.dropna(axis=1, how='all', inplace=True)
        df_raw.columns = df_raw.iloc[0]
        df = df_raw[1:].copy()
        df['Vintage'] = year
        # Date handling
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        else:
            date_input = st.sidebar.date_input(f"Collection date for {year}")
            df['Date'] = pd.to_datetime(date_input)
        # Convert metrics
        for col in ['Brix', 'pH', 'TA', 'MA']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        all_data.append(df)
    df_all = pd.concat(all_data, ignore_index=True)
    # Filters
    st.sidebar.header("Filters")
    vintages = sorted(df_all['Vintage'].unique())
    selected_vintages = st.sidebar.multiselect("Vintages", vintages, default=vintages)
    variety = st.sidebar.selectbox("Variety", sorted(df_all['Variety'].dropna().unique()))
    block = st.sidebar.selectbox("Block", sorted(df_all['Block'].dropna().unique()))
    metric = st.sidebar.selectbox("Metric", ['Brix', 'pH', 'TA', 'MA'])
    filtered = df_all[
        (df_all['Vintage'].isin(selected_vintages)) &
        (df_all['Variety'] == variety) &
        (df_all['Block'] == block)
    ].sort_values('Date')
    # Plot comparison
    st.subheader(f"{metric} Comparison")
    for vintage in selected_vintages:
        grp = filtered[filtered['Vintage'] == vintage]
        if not grp.empty:
            st.line_chart(grp.set_index('Date')[metric], width=700, height=300)
    # Prediction
    st.subheader("Readiness Prediction")
    grp = filtered.dropna(subset=[metric])
    if len(grp) > 1:
        X = np.arange(len(grp)).reshape(-1,1)
        y = grp[metric].values
        model = LinearRegression().fit(X, y)
        future = np.arange(len(grp), len(grp)+5).reshape(-1,1)
        preds = model.predict(future)
        dates = list(grp['Date']) + [grp['Date'].max() + pd.Timedelta(days=7*(i+1)) for i in range(5)]
        df_pred = pd.DataFrame({metric: np.concatenate([y, preds])}, index=dates)
        st.line_chart(df_pred, width=700, height=300)
        # Check readiness
        ready = np.where(preds >= brix_thresh)[0]
        if ready.size > 0:
            date_ready = dates[len(grp) + ready[0]]
            st.success(f"Predicted readiness around {date_ready.date()}")
        else:
            st.warning("No readiness predicted in forecast window.")
    # Summary
    st.subheader("Summary")
    summary = filtered.groupby('Vintage')[[ 'Brix','pH','TA','MA']].mean().round(2)
    st.dataframe(summary)