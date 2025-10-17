# WFP Ethiopia – Targeting Verification Sampling App (Somali Region RAM)
# Author: M365 Copilot for MMussie (WFP ET Somali Region RAM)
# Updated: 2025-10-17
# v3 FULL BUILD:
#  - Keeps ALL original fields in outputs
#  - UNIQUE households only (drop dup HH_ID) and sampling WITHOUT replacement
#  - Reallocate village shortfalls within kebele to other selected villages (capacity-aware)
#  - NumPy dtype fix: use pandas-native assignment for Group to avoid DTypePromotionError
#  - Adds quick self-check of group counts & mapping preview

import io, os
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='WFP ET Verification Sampling – Somali RAM', layout='wide')

# ---- Header ----
col_logo, col_title = st.columns([1,4])
with col_logo:
    logo_path = os.path.join(os.path.dirname(__file__), 'assets', 'wfp_logo_placeholder.png')
    if os.path.exists(logo_path):
        st.image(logo_path, width=96)
with col_title:
    st.title('Targeting Verification – Sampling App')
    st.caption('Somali Region RAM · PPS village selection and systematic HH sampling per ETCO Verification SOP (Sept 2024)')

# ---- Sidebar configuration ----
with st.sidebar:
    st.header('Configuration')
    seed = st.number_input('Random seed', value=20241017, step=1)
    mode = st.selectbox('Village selection mode', ['Auto (SOP rule)', 'Manual (fixed number per kebele)'])
    if mode.startswith('Manual'):
        manual_villages_to_sample = st.number_input('Villages (EAs) per kebele', min_value=1, max_value=50, value=2, step=1)
        st.caption('SOP default is 2; consider 4–6 if kebele has ≥7 villages.')
    else:
        st.info('Auto rule: <7→2; 7–9→4; 10–12→5; ≥13→6 villages per kebele')
    st.markdown('---')

    per_group_sample = st.number_input('Sample size per group per kebele', min_value=1, max_value=2000, value=30, step=1)
    reserve_count = st.number_input('Reserve households per group per village (optional)', min_value=0, max_value=500, value=0, step=1)
    st.info('Sampling is without replacement (unique HHs). Village shortfalls are reallocated within the kebele.')

# ---- 1) Upload ----
st.markdown('### 1) Upload Household Listing')
uploaded = st.file_uploader('Upload Excel/CSV (household rows; first sheet used if Excel)', type=['xlsx','csv'])
if uploaded is None:
    st.stop()

# Load file
if uploaded.name.lower().endswith('.csv'):
    df = pd.read_csv(uploaded)
else:
    xl = pd.ExcelFile(uploaded)
    df = pd.read_excel(xl, sheet_name=xl.sheet_names[0])

orig_cols = list(df.columns)

# ---- 2) Map columns ----
st.markdown('### 2) Map Columns')
cols = list(df.columns)
col_kebele = st.selectbox('Kebele column', cols)
col_village = st.selectbox('Village / EA / Block column', cols, index=min(1, len(cols)-1))
col_elig = st.selectbox('Eligibility flag column', cols)
col_hhid_choice = st.selectbox('Household ID column (optional)', ['<None>'] + cols)

# HH_ID handling
if col_hhid_choice == '<None>':
    df = df.reset_index(drop=False).rename(columns={'index':'__row__'})
    df['HH_ID'] = (
        df[col_kebele].astype(str).str.strip() + '|' +
        df[col_village].astype(str).str.strip() + '|' +
        df['__row__'].astype(str)
    )
    synthetic_id = True
else:
    df['HH_ID'] = df[col_hhid_choice].astype(str)
    synthetic_id = False

# ---- 3) Eligibility mapping ----
st.markdown('### 3) Define Eligibility Mapping')
eligible_values = st.text_input('Eligible values (comma-separated)', 'Eligible,Yes,1,True')
noneligible_values = st.text_input('Non-eligible values (comma-separated)', 'Non-eligible,No,0,False,Ineligible')
eligible_set = set(v.strip().lower() for v in eligible_values.split(',') if v.strip())
noneligible_set = set(v.strip().lower() for v in noneligible_values.split(',') if v.strip())

work = df.copy()
# Unique-only: drop dup HH_ID
work = work.drop_duplicates(subset=['HH_ID']).copy()
# Normalize
work['__elig_val__'] = work[col_elig].astype(str).str.strip().str.lower()
# Safe pandas assignment to avoid NumPy dtype issues
work['Group'] = pd.Series(pd.NA, index=work.index, dtype='string')
work.loc[work['__elig_val__'].isin(eligible_set), 'Group'] = 'Eligible'
work.loc[work['__elig_val__'].isin(noneligible_set), 'Group'] = 'Non-eligible'

# Self-check panel
c1, c2, c3 = st.columns(3)
with c1:
    st.metric('Eligible (mapped)', int((work['Group']=='Eligible').sum()))
with c2:
    st.metric('Non-eligible (mapped)', int((work['Group']=='Non-eligible').sum()))
with c3:
    st.metric('Unmapped / invalid', int(work['Group'].isna().sum()))

if work['Group'].isna().any():
    st.error('Some rows are neither Eligible nor Non‑eligible by your mapping. Adjust the values above to proceed.')
    st.stop()

# Optional accessibility filter
a_cols = ['<None>'] + [c for c in cols if c != col_elig]
accessible_col = st.selectbox('Accessibility column (optional)', a_cols)
accessible_true_values = st.text_input('Values that mean Accessible (comma-separated)', '1,True,Yes,Accessible')
accessible_true_set = set(v.strip().lower() for v in accessible_true_values.split(',') if v.strip())
if accessible_col != '<None>':
    mask_access = work[accessible_col].astype(str).str.strip().str.lower().isin(accessible_true_set)
    removed = int((~mask_access).sum())
    work = work[mask_access].copy()
    st.warning(f'Excluded {removed} records not marked as Accessible.')

# ---- Helpers ----

def auto_k_for_kebele(n_villages:int) -> int:
    if n_villages < 7: return 2
    if 7 <= n_villages <= 9: return 4
    if 10 <= n_villages <= 12: return 5
    return 6


def select_villages_pps(kebele_df: pd.DataFrame, k: int, rng) -> list:
    counts = kebele_df.groupby(col_village).size().reset_index(name='N')
    counts = counts[counts['N']>0]
    if counts.empty:
        return []
    probs = counts['N'].to_numpy(dtype=float)
    probs = probs / probs.sum()
    k = min(k, len(counts))
    idx = rng.choice(len(counts), size=k, replace=False, p=probs)
    chosen = counts.iloc[idx].sort_values(by='N', ascending=False)
    return chosen[col_village].tolist()


def rebalance_allocation(selected_villages, df_k, group_name, initial_alloc):
    """Reallocate village shortfalls within kebele using available capacity for `group_name`."""
    caps = []
    for v in selected_villages:
        caps.append(int(((df_k[col_village] == v) & (df_k['Group'] == group_name)).sum()))
    alloc = [min(initial_alloc[i], caps[i]) for i in range(len(selected_villages))]
    deficit = sum(initial_alloc) - sum(alloc)
    # Greedy, largest remaining capacity first
    while deficit > 0:
        rem = [max(0, caps[i] - alloc[i]) for i in range(len(selected_villages))]
        if sum(rem) == 0:
            break
        order = sorted(range(len(selected_villages)), key=lambda i: rem[i], reverse=True)
        for i in order:
            if rem[i] <= 0:
                continue
            alloc[i] += 1
            deficit -= 1
            if deficit == 0:
                break
    return alloc, caps


def systematic_without_replacement(df_list: pd.DataFrame, take: int, rng):
    N = len(df_list)
    if take <= 0 or N == 0:
        return df_list.iloc[0:0].copy(), {'SI': None, 'RS': None, 'Method': 'Empty'}
    if take >= N:
        return df_list.copy(), {'SI': None, 'RS': None, 'Method': 'All (N < n) – unique only'}
    from math import floor
    si = max(1, floor(N / take))
    rs = int(rng.integers(1, si+1))
    positions, pos = [], rs
    while len(positions) < take:
        if pos <= N: positions.append(pos)
        else:
            pos -= N
            if pos <= N: positions.append(pos)
        pos += si
    idx = [p-1 for p in positions]
    chosen = df_list.iloc[idx].copy()
    return chosen, {'SI': si, 'RS': rs, 'Method': 'Systematic (no replacement)'}

# ---- 4) Run ----
st.markdown('### 4) Run PPS + Systematic Sampling')
if st.button('Run PPS + Systematic Sampling'):
    rng = np.random.default_rng(int(seed))
    village_selection_rows, hh_rows, kebele_summary_rows = [], [], []

    for kebele, df_k in work.groupby(col_kebele):
        total_villages = int(df_k[col_village].nunique())
        if mode.startswith('Auto'):
            k_vill = auto_k_for_kebele(total_villages)
            rule_used = f'Auto: {k_vill} (total villages={total_villages})'
        else:
            k_vill = int(manual_villages_to_sample)
            rule_used = f'Manual: {k_vill} (total villages={total_villages})'

        selected_villages = select_villages_pps(df_k, k_vill, rng)
        if not selected_villages:
            continue

        groups = ['Eligible','Non-eligible']
        per_group = int(per_group_sample)

        # Split evenly then rebalance per group by village capacity
        group_allocs = {}
        for g in groups:
            initial = [per_group // len(selected_villages)] * len(selected_villages)
            rem = per_group - sum(initial)
            for t in range(rem):
                initial[t % len(selected_villages)] += 1
            rebalanced, caps = rebalance_allocation(selected_villages, df_k, g, initial)
            group_allocs[g] = rebalanced

        # Log per-village plan after rebalance
        for i, v in enumerate(selected_villages):
            total_in_v = int((df_k[col_village] == v).sum())
            village_selection_rows.append({
                'Kebele': kebele,
                'Selected_Village': v,
                'Total_HHs_in_Village': total_in_v,
                'Planned_Sample_Eligible': group_allocs['Eligible'][i],
                'Planned_Sample_NonEligible': group_allocs['Non-eligible'][i],
                'Village_Selection_Mode': rule_used
            })

        # Sample (unique only, without replacement)
        for i, v in enumerate(selected_villages):
            df_v = df_k[df_k[col_village] == v]
            for g in groups:
                df_list = df_v[df_v['Group']==g].drop_duplicates(subset=['HH_ID']).reset_index(drop=True)
                needed = int(group_allocs[g][i])
                take = min(needed, len(df_list))
                chosen, meta = systematic_without_replacement(df_list, take, rng)

                chosen_orig = chosen[orig_cols].copy()
                chosen_orig = chosen_orig.assign(
                    Kebele=kebele,
                    Village=v,
                    Group_Sampled=g,
                    SI=meta['SI'], RS=meta['RS'], Method=meta['Method']
                )
                chosen_orig['Selection_Order'] = range(1, len(chosen_orig)+1)
                hh_rows.append(chosen_orig)

        # Kebele summary (planned vs actual)
        tmp = pd.DataFrame(hh_rows)
        if not tmp.empty:
            tmp_k = tmp[tmp['Kebele']==kebele]
            act_el = int((tmp_k['Group_Sampled']=='Eligible').sum())
            act_ne = int((tmp_k['Group_Sampled']=='Non-eligible').sum())
        else:
            act_el = act_ne = 0
        kebele_summary_rows.append({
            'Kebele': kebele,
            'Total_Villages_in_Kebele': total_villages,
            'Villages_Selected': len(selected_villages),
            'Villages_List': ', '.join(map(str, selected_villages)),
            'Village_Selection_Mode': rule_used,
            'Planned_Eligible_Total': per_group,
            'Planned_NonEligible_Total': per_group,
            'Actual_Eligible_Total': act_el,
            'Actual_NonEligible_Total': act_ne,
            'Shortfall_Eligible': max(0, per_group - act_el),
            'Shortfall_NonEligible': max(0, per_group - act_ne)
        })

    if not hh_rows:
        st.error('No households were sampled. Check mappings and input data.')
        st.stop()

    # Build outputs
    village_sel_df = pd.DataFrame(village_selection_rows)
    kebele_summary_df = pd.DataFrame(kebele_summary_rows)
    sampled_df = pd.concat(hh_rows, ignore_index=True)

    # Optional reserve list (unique only; exclude sampled)
    reserve_df = None
    if reserve_count > 0:
        reserve_rows = []
        rng2 = np.random.default_rng(int(seed) + 1)
        sampled_keys = set(zip(sampled_df['Kebele'], sampled_df['Village'], sampled_df['Group_Sampled'], sampled_df.get('HH_ID', pd.Series([None]*len(sampled_df)))))
        for (k, v, g), sub in work.groupby([col_kebele, col_village, 'Group']):
            if not village_sel_df.empty:
                valid_vs = set(village_sel_df.loc[village_sel_df['Kebele']==k, 'Selected_Village'])
                if v not in valid_vs:
                    continue
            pool = sub.drop_duplicates(subset=['HH_ID']).copy()
            pool['__key__'] = list(zip([k]*len(pool), [v]*len(pool), pool['Group'], pool['HH_ID']))
            pool = pool[~pool['__key__'].isin(sampled_keys)].reset_index(drop=True)
            if len(pool)==0:
                continue
            take = min(int(reserve_count), len(pool))
            idx = rng2.choice(range(len(pool)), size=take, replace=False)
            chosen_r = pool.iloc[idx].copy()
            chosen_r = chosen_r[orig_cols].assign(Kebele=k, Village=v, Group_Sampled=g)
            reserve_rows.append(chosen_r)
        if reserve_rows:
            reserve_df = pd.concat(reserve_rows, ignore_index=True)

    # Write Excel output
    out_buf = io.BytesIO()
    with pd.ExcelWriter(out_buf, engine='openpyxl') as xw:
        kebele_summary_df.to_excel(xw, index=False, sheet_name='Summary_By_Kebele')
        village_sel_df.to_excel(xw, index=False, sheet_name='Village_Selection')
        sampled_df.to_excel(xw, index=False, sheet_name='Household_Samples_AllFields')
        if reserve_df is not None:
            reserve_df.to_excel(xw, index=False, sheet_name='Reserve_List_AllFields')
        pd.DataFrame({
            'Parameter': ['Seed','Mode','Per_Group_Sample','UniqueOnly','Synthetic_HH_ID','Reserve_Per_GroupPerVillage'],
            'Value': [seed, mode, per_group_sample, True, synthetic_id, reserve_count]
        }).to_excel(xw, index=False, sheet_name='Run_Parameters')

    st.success('Sampling complete. Download your outputs below.')
    st.download_button('Download sampled results (Excel – full fields)', data=out_buf.getvalue(), file_name='verification_sampling_output_fullfields.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    # Optional CSV bundle download
    import zipfile
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('Summary_By_Kebele.csv', kebele_summary_df.to_csv(index=False))
        zf.writestr('Village_Selection.csv', village_sel_df.to_csv(index=False))
        zf.writestr('Household_Samples_AllFields.csv', sampled_df.to_csv(index=False))
        if reserve_df is not None:
            zf.writestr('Reserve_List_AllFields.csv', reserve_df.to_csv(index=False))
    st.download_button('Download outputs (CSV bundle .zip)', data=zip_buf.getvalue(), file_name='verification_sampling_outputs_csv.zip', mime='application/zip')
