# WFP ET – Targeting Verification Sampling App (Somali RAM)

**This build adds reallocation + uniqueness:**
- Preserves **all original fields** in `Household_Samples_AllFields`.
- Enforces **unique households** (drops duplicates by `HH_ID`, samples **without replacement**).
- If a village has fewer HHs than its target, the **shortfall is reallocated** to other selected villages within the **same kebele**.

Implements PPS village selection and systematic HH sampling per ETCO Verification SOP (Sept 2024). Default per-kebele: **30 Eligible + 30 Non‑eligible**. Auto villages: `<7→2; 7–9→4; 10–12→5; ≥13→6`.

## Run
```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
