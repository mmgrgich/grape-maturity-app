import streamlit as st
import pandas as pd
import numpy as np
import datetime
import re
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
        if 'Vineyard' in df.columns and 'Block' in df.columns:
            df['Vineyard'] = df['Vineyard'].ffill()
        if df.columns.duplicated().any():
            st.error(
                f"Duplicate columns found in file {file.name}: "
                f"{df.columns[df.columns.duplicated()].tolist()}. "
                "Please fix the Excel file to remove duplicate column names."
            )
            continue
        df['Vintage'] = year

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y', errors='coerce')
            df['Date'] = df['Date'].fillna(pd.to_datetime(df['Date'], errors='coerce'))
        else:
            filename = file.name
            match = re.search(r'(\d{1,2})-(\d{1,2})-(\d{2})(?:\.xlsx)?$', filename)
            parsed_date = None
            if match:
                month, day, year2 = match.groups()
                year4 = int(year2)
                year4 += 2000 if year4 < 100 else 0
                date_str = f"{month}/{day}/{year4}"
                parsed_date = pd.to_datetime(date_str, format='%m/%d/%Y', errors='coerce')
            if parsed_date is not None and not pd.isnull(parsed_date):
                df['Date'] = parsed_date
                st.toast(f"Applied date {parsed_date.strftime('%Y-%m-%d')} from file name to all rows in {file.name}", icon="✅")
            else:
                st.warning(
                    f"File '{file.name}' is missing a 'Date' column and no recognizable date was found in the file name. "
                    f"Please enter the correct collection date for this file below or update your Excel format.",
                    icon="⚠️"
                )
                date_input = st.sidebar.date_input(
                    f"Collection date for {year} ({file.name}) [REQUIRED]",
                    value=None,
                    key=f"date_input_{i}_{file.name}"
                )
                if date_input is None:
                    st.stop()
                df['Date'] = pd.to_datetime(date_input)

        all_data.append(df)
    if not all_data:
        st.error("No valid data could be loaded from the uploaded files.")
    else:
        df_all = pd.concat(all_data, ignore_index=True)
        df_all.columns = df_all.columns.str.strip()
        for col in ['Brix', 'pH', 'TA', 'MA']:
            if col in df_all.columns:
                df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

        # --- FILTERING & FLIP-THROUGH FUNCTIONALITY (Now with Variety) ---
        st.sidebar.header("Vineyard, Block, and Variety Navigation")

        vineyards = sorted(df_all['Vineyard'].dropna().astype(str).unique())
        if "vineyard_select" not in st.session_state:
            st.session_state["vineyard_select"] = vineyards[0]
        vineyard = st.session_state["vineyard_select"]

        blocks = sorted(df_all[df_all['Vineyard'].astype(str) == vineyard]['Block'].dropna().astype(str).unique())
        if "block_select" not in st.session_state or st.session_state["block_select"] not in blocks:
            st.session_state["block_select"] = blocks[0]
        block = st.session_state["block_select"]

        if 'Variety' in df_all.columns:
            varieties = sorted(df_all[
                (df_all['Vineyard'].astype(str) == vineyard) &
                (df_all['Block'].astype(str) == block)
            ]['Variety'].dropna().astype(str).unique())
            if not varieties:
                varieties = ['']
            if "variety_select" not in st.session_state or st.session_state["variety_select"] not in varieties:
                st.session_state["variety_select"] = varieties[0]
            variety = st.session_state["variety_select"]
        else:
            varieties = ['']
            variety = ''

        vineyard = st.sidebar.selectbox("Vineyard", vineyards, index=vineyards.index(vineyard), key="vineyard_select")
        blocks = sorted(df_all[df_all['Vineyard'].astype(str) == vineyard]['Block'].dropna().astype(str).unique())
        block = st.sidebar.selectbox("Block", blocks, index=blocks.index(block) if block in blocks else 0, key="block_select")
        if 'Variety' in df_all.columns:
            varieties = sorted(df_all[
                (df_all['Vineyard'].astype(str) == vineyard) &
                (df_all['Block'].astype(str) == block)
            ]['Variety'].dropna().astype(str).unique())
            if not varieties:
                varieties = ['']
            variety = st.sidebar.selectbox("Variety", varieties, index=varieties.index(variety) if variety in varieties else 0, key="variety_select")
        else:
            variety = ''

        if vineyard != st.session_state["vineyard_select"]:
            st.session_state["vineyard_select"] = vineyard
            new_blocks = sorted(df_all[df_all['Vineyard'].astype(str) == vineyard]['Block'].dropna().astype(str).unique())
            st.session_state["block_select"] = new_blocks[0] if new_blocks else None
            if 'Variety' in df_all.columns:
                new_varieties = sorted(df_all[
                    (df_all['Vineyard'].astype(str) == vineyard) &
                    (df_all['Block'].astype(str) == st.session_state["block_select"])
                ]['Variety'].dropna().astype(str).unique())
                st.session_state["variety_select"] = new_varieties[0] if new_varieties else ''
        if block != st.session_state["block_select"]:
            st.session_state["block_select"] = block
            if 'Variety' in df_all.columns:
                new_varieties = sorted(df_all[
                    (df_all['Vineyard'].astype(str) == vineyard) &
                    (df_all['Block'].astype(str) == block)
                ]['Variety'].dropna().astype(str).unique())
                st.session_state["variety_select"] = new_varieties[0] if new_varieties else ''
        if 'Variety' in df_all.columns and variety != st.session_state["variety_select"]:
            st.session_state["variety_select"] = variety

        v_idx = vineyards.index(st.session_state["vineyard_select"])
        blocks = sorted(df_all[df_all['Vineyard'].astype(str) == st.session_state["vineyard_select"]]['Block'].dropna().astype(str).unique())
        b_idx = blocks.index(st.session_state["block_select"]) if st.session_state["block_select"] in blocks else 0
        if 'Variety' in df_all.columns:
            varieties = sorted(df_all[
                (df_all['Vineyard'].astype(str) == st.session_state["vineyard_select"]) &
                (df_all['Block'].astype(str) == st.session_state["block_select"])
            ]['Variety'].dropna().astype(str).unique())
            if not varieties:
                varieties = ['']
            var_idx = varieties.index(st.session_state["variety_select"]) if st.session_state["variety_select"] in varieties else 0
        else:
            varieties = ['']
            var_idx = 0
        col1, col2, col3, col4, col5, col6 = st.sidebar.columns([1,1,1,1,1,1])
        with col1:
            if st.button("Prev Vineyard"):
                if v_idx > 0:
                    st.session_state["vineyard_select"] = vineyards[v_idx - 1]
                    new_blocks = sorted(df_all[df_all['Vineyard'].astype(str) == vineyards[v_idx - 1]]['Block'].dropna().astype(str).unique())
                    st.session_state["block_select"] = new_blocks[0] if new_blocks else None
                    if 'Variety' in df_all.columns:
                        new_varieties = sorted(df_all[
                            (df_all['Vineyard'].astype(str) == vineyards[v_idx - 1]) &
                            (df_all['Block'].astype(str) == st.session_state["block_select"])
                        ]['Variety'].dropna().astype(str).unique())
                        st.session_state["variety_select"] = new_varieties[0] if new_varieties else ''
        with col2:
            if st.button("Next Vineyard"):
                if v_idx < len(vineyards)-1:
                    st.session_state["vineyard_select"] = vineyards[v_idx + 1]
                    new_blocks = sorted(df_all[df_all['Vineyard'].astype(str) == vineyards[v_idx + 1]]['Block'].dropna().astype(str).unique())
                    st.session_state["block_select"] = new_blocks[0] if new_blocks else None
                    if 'Variety' in df_all.columns:
                        new_varieties = sorted(df_all[
                            (df_all['Vineyard'].astype(str) == vineyards[v_idx + 1]) &
                            (df_all['Block'].astype(str) == st.session_state["block_select"])
                        ]['Variety'].dropna().astype(str).unique())
                        st.session_state["variety_select"] = new_varieties[0] if new_varieties else ''
        with col3:
            if st.button("Prev Block"):
                if b_idx > 0:
                    st.session_state["block_select"] = blocks[b_idx - 1]
                    if 'Variety' in df_all.columns:
                        new_varieties = sorted(df_all[
                            (df_all['Vineyard'].astype(str) == st.session_state["vineyard_select"]) &
                            (df_all['Block'].astype(str) == blocks[b_idx - 1])
                        ]['Variety'].dropna().astype(str).unique())
                        st.session_state["variety_select"] = new_varieties[0] if new_varieties else ''
        with col4:
            if st.button("Next Block"):
                if b_idx < len(blocks)-1:
                    st.session_state["block_select"] = blocks[b_idx + 1]
                    if 'Variety' in df_all.columns:
                        new_varieties = sorted(df_all[
                            (df_all['Vineyard'].astype(str) == st.session_state["vineyard_select"]) &
                            (df_all['Block'].astype(str) == blocks[b_idx + 1])
                        ]['Variety'].dropna().astype(str).unique())
                        st.session_state["variety_select"] = new_varieties[0] if new_varieties else ''
        if 'Variety' in df_all.columns:
            with col5:
                if st.button("Prev Variety"):
                    if var_idx > 0:
                        st.session_state["variety_select"] = varieties[var_idx - 1]
            with col6:
                if st.button("Next Variety"):
                    if var_idx < len(varieties) - 1:
                        st.session_state["variety_select"] = varieties[var_idx + 1]

        available_metrics = [col for col in ['Brix', 'pH', 'TA', 'MA'] if col in df_all.columns]
        metric = st.sidebar.selectbox("Metric", available_metrics)

        filtered = df_all[
            (df_all['Vineyard'].astype(str) == st.session_state["vineyard_select"]) &
            (df_all['Block'].astype(str) == st.session_state["block_select"])
        ].copy()
        if 'Variety' in df_all.columns:
            filtered = filtered[filtered['Variety'].astype(str) == st.session_state["variety_select"]]

        # ---- 1. PREDICTION: Only current calendar year ----
        title_variety = f" ({st.session_state['variety_select']})" if st.session_state.get("variety_select", '') else ''
        st.subheader(f"Prediction for {st.session_state['vineyard_select']} Block {st.session_state['block_select']}{title_variety} in {datetime.datetime.now().year}")
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

        # ---- 2. AGGREGATE PLOTS: All vintages for this block/vineyard/variety (month-only x axis, with linear regression) ----
        st.subheader(f"All Vintages: {st.session_state['vineyard_select']} Block {st.session_state['block_select']}{title_variety} ({metric})")

        fig, ax = plt.subplots(figsize=(9,5))

        mask_grow = filtered['Date'].dt.month.between(6,12)
        filtered_grow = filtered[mask_grow].copy()

        dummy_year = 2000
        filtered_grow['PlotDate'] = filtered_grow['Date'].apply(lambda d: d.replace(year=dummy_year))

        vintages = sorted(filtered_grow['Vintage'].dropna().unique())
        colors = plt.cm.get_cmap('tab10', len(vintages))

        for idx, vtg in enumerate(vintages):
            grp = filtered_grow[filtered_grow['Vintage'] == vtg].dropna(subset=['PlotDate', metric])
            if not grp.empty:
                grp = grp.sort_values('PlotDate')
                ax.plot(grp['PlotDate'], grp[metric], marker='o', label=str(vtg), color=colors(idx))
                X = mdates.date2num(grp['PlotDate']).reshape(-1, 1)
                y = grp[metric].values
                if len(X) > 1:
                    model = LinearRegression().fit(X, y)
                    x_fit = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
                    y_fit = model.predict(x_fit)
                    ax.plot(mdates.num2date(x_fit.flatten()), y_fit, color=colors(idx), linestyle='--', alpha=0.7)

        x_start = pd.Timestamp(f"{dummy_year}-06-01")
        x_end = pd.Timestamp(f"{dummy_year}-12-31")
        ax.set_xlim(x_start, x_end)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        ax.set_title(f"All Vintages for {st.session_state['vineyard_select']} Block {st.session_state['block_select']}{title_variety} ({metric})")
        ax.set_xlabel("Month")
        ax.set_ylabel(metric)
        ax.legend(title="Vintage", loc="best", fontsize='small')
        plt.tight_layout()
        st.pyplot(fig)

        # ---- 3. SUMMARY (by Vineyard, Block, Variety, all years, with best fit line) ----
        st.subheader("Summary (by Vineyard, Block, Variety, all vintages)")
        summary_cols = [col for col in ['Brix', 'pH', 'TA', 'MA'] if col in filtered.columns]
        group_cols = ['Vineyard', 'Block']
        if 'Variety' in filtered.columns:
            group_cols.append('Variety')
        if summary_cols:
            summary = filtered.groupby(group_cols)[summary_cols].mean(numeric_only=True).round(2)
            st.dataframe(summary)

            # For the selected metric: plot mean value by vintage and add best-fit line
            if 'Vintage' in filtered.columns:
                st.subheader(f"Mean {metric} by Vintage for {st.session_state['vineyard_select']} Block {st.session_state['block_select']}{title_variety}")
                means = filtered.groupby('Vintage')[metric].mean().dropna()
                vintages = pd.to_numeric(means.index)
                values = means.values
                fig2, ax2 = plt.subplots(figsize=(7,4))
                ax2.scatter(vintages, values, color='tab:blue', label='Mean')
                if len(vintages) > 1:
                    reg = LinearRegression().fit(vintages.values.reshape(-1,1), values)
                    y_pred = reg.predict(np.array([vintages.min(), vintages.max()]).reshape(-1,1))
                    ax2.plot([vintages.min(), vintages.max()], y_pred, color='tab:red', linestyle='--', label='Best fit')
                ax2.set_xlabel("Vintage")
                ax2.set_ylabel(f"Mean {metric}")
                ax2.set_title(f"Mean {metric} by Vintage")
                ax2.legend()
                st.pyplot(fig2)
        else:
            st.info("No summary metrics available for the selected data.")
