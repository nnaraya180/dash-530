
import re
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------- Page config ----------
st.set_page_config(page_title="MIMIC-IV Procedures + Admissions Dashboard", layout="wide")

# ---------- Helpers ----------
def load_csv(source, name: str):
    # Load CSV from an uploaded file or from a Path string; return None on failure with a warning.
    try:
        if source is None:
            return None
        if hasattr(source, "read"):  # UploadedFile
            return pd.read_csv(source)
        p = Path(source)
        if p.exists():
            return pd.read_csv(p)
    except Exception as e:
        st.warning(f"Could not load {name}: {e}")
    return None

def coerce_datetime(series):
    return pd.to_datetime(series, errors="coerce")

STOPWORDS = set('a an the and or for of to in with without by on at from via over under during after before within among between '
                'unspecified other others type procedure surgical surgery approach device left right bilateral open percutaneous '
                'revision initial removal insertion replacement repair unspecified, unspecified. unspecified-'.split())

def tokenize_titles(titles: pd.Series, top_n=30):
    tokens = []
    for title in titles.dropna().astype(str):
        # Normalize punctuation -> space, lower
        text = re.sub(r"[^a-zA-Z0-9 ]+", " ", title.lower())
        parts = [p for p in text.split() if len(p) > 2 and p not in STOPWORDS]
        tokens.extend(parts)
    counts = Counter(tokens)
    return pd.DataFrame(counts.most_common(top_n), columns=["token", "count"])

def assign_age_group(age_series):
    bins = [0, 18, 30, 45, 60, 75, 90, 200]
    labels = ["0–17", "18–29", "30–44", "45–59", "60–74", "75–89", "90+"]
    return pd.cut(age_series, bins=bins, labels=labels, right=False, include_lowest=True)

# ---------- Sidebar: Data selection ----------
st.sidebar.header("Data sources")

use_uploads = st.sidebar.toggle("Upload CSVs here (otherwise, use ./data/)", value=False)

if use_uploads:
    patients_up = st.sidebar.file_uploader("patients.csv", type=["csv"])
    procedures_up = st.sidebar.file_uploader("procedures_icd.csv", type=["csv"])
    dict_up = st.sidebar.file_uploader("d_icd_procedures.csv", type=["csv"])
    admissions_up = st.sidebar.file_uploader("admissions.csv", type=["csv"])
    patients = load_csv(patients_up, "patients.csv")
    procedures = load_csv(procedures_up, "procedures_icd.csv")
    d_proc = load_csv(dict_up, "d_icd_procedures.csv")
    admissions = load_csv(admissions_up, "admissions.csv")
else:
    data_dir = Path("data")
    patients = load_csv(data_dir / "patients.csv", "patients.csv")
    procedures = load_csv(data_dir / "procedures_icd.csv", "procedures_icd.csv")
    d_proc = load_csv(data_dir / "d_icd_procedures.csv", "d_icd_procedures.csv")
    admissions = load_csv(data_dir / "admissions.csv", "admissions.csv")

# Require the three base tables; admissions is optional but unlocks extra pages
base_loaded = all([isinstance(patients, pd.DataFrame), isinstance(procedures, pd.DataFrame), isinstance(d_proc, pd.DataFrame)])
if not base_loaded:
    st.error("Please provide patients.csv, procedures_icd.csv, and d_icd_procedures.csv (via uploads or ./data/).")
    st.stop()

# ---------- Light cleaning / joins ----------
# Standardize column names
patients.columns = [c.lower() for c in patients.columns]
procedures.columns = [c.lower() for c in procedures.columns]
d_proc.columns = [c.lower() for c in d_proc.columns]

if isinstance(admissions, pd.DataFrame):
    admissions.columns = [c.lower() for c in admissions.columns]

# Coerce date in procedures
if "chartdate" in procedures.columns:
    procedures["chartdate"] = coerce_datetime(procedures["chartdate"])
    procedures["year"] = procedures["chartdate"].dt.year
    procedures["month"] = procedures["chartdate"].dt.to_period("M").astype(str)
else:
    procedures["year"] = np.nan
    procedures["month"] = np.nan

# Merge long titles onto procedures
proc_merged = procedures.merge(d_proc, on=["icd_code", "icd_version"], how="left")
proc_merged.rename(columns={"long_title": "procedure_title"}, inplace=True)

# Derive per-patient and per-admission procedure counts
per_patient_counts = proc_merged.groupby("subject_id", as_index=False).size().rename(columns={"size": "procedures_per_patient"})
patients_enriched = patients.merge(per_patient_counts, on="subject_id", how="left")
patients_enriched["procedures_per_patient"] = patients_enriched["procedures_per_patient"].fillna(0).astype(int)

per_admit_counts = proc_merged.groupby(["subject_id", "hadm_id"], as_index=False).size().rename(columns={"size": "procedures_per_admission"})

# Admissions enrichment (if available)
if isinstance(admissions, pd.DataFrame):
    # Parse datetimes
    for col in ["admittime", "dischtime", "deathtime"]:
        if col in admissions.columns:
            admissions[col] = coerce_datetime(admissions[col])

    # LOS (days)
    if all(c in admissions.columns for c in ["admittime", "dischtime"]):
        los = (admissions["dischtime"] - admissions["admittime"]).dt.total_seconds() / 86400.0
        admissions["los_days"] = los
        admissions["admit_month"] = admissions["admittime"].dt.to_period("M").astype(str)
        admissions["admit_year"] = admissions["admittime"].dt.year
    else:
        admissions["los_days"] = np.nan
        admissions["admit_month"] = np.nan
        admissions["admit_year"] = np.nan

    # In-hospital mortality
    admissions["died_in_hosp"] = admissions.get("deathtime", pd.Series([np.nan]*len(admissions))).notna()

    # Join procedures per admission
    admissions_enriched = admissions.merge(per_admit_counts, on=["subject_id", "hadm_id"], how="left")
    admissions_enriched["procedures_per_admission"] = admissions_enriched["procedures_per_admission"].fillna(0).astype(int)
    # Rate normalized by LOS (procedures per LOS day)
    admissions_enriched["proc_per_day"] = admissions_enriched.apply(
        lambda r: r["procedures_per_admission"]/r["los_days"] if pd.notna(r["los_days"]) and r["los_days"] > 0 else np.nan,
        axis=1
    )
else:
    admissions_enriched = None

# Age group
if "anchor_age" in patients_enriched.columns:
    patients_enriched["age_group"] = assign_age_group(patients_enriched["anchor_age"])
else:
    patients_enriched["age_group"] = pd.NA

# ---------- High-level metrics ----------
n_patients = int(patients_enriched["subject_id"].nunique())
n_procedures = int(proc_merged.shape[0])
n_codes = int(proc_merged["icd_code"].nunique())
median_age = float(patients_enriched["anchor_age"].median()) if "anchor_age" in patients_enriched.columns else float("nan")
gender_counts = patients_enriched["gender"].value_counts(dropna=False).rename_axis("gender").reset_index(name="count") if "gender" in patients_enriched.columns else pd.DataFrame(columns=["gender","count"])

if isinstance(admissions_enriched, pd.DataFrame):
    n_admissions = int(admissions_enriched["hadm_id"].nunique())
    median_los = float(admissions_enriched["los_days"].median(skipna=True))
    mort_rate = float((admissions_enriched["died_in_hosp"] == True).mean()) if "died_in_hosp" in admissions_enriched.columns else float("nan")
else:
    n_admissions = 0
    median_los = float("nan")
    mort_rate = float("nan")

# ---------- Navigation ----------
tab_names = ["Overview", "Procedures", "Patient Profiles", "Qualitative Themes"]
if isinstance(admissions_enriched, pd.DataFrame):
    tab_names.append("Admissions & Outcomes")
tab_names.append("Findings & Risks")

st.title("MIMIC-IV Procedures + Admissions Dashboard")
st.caption("Audience: domain stakeholders; visuals emphasize interpretation over statistical technicalities.")

pages = st.tabs(tab_names)

# ---------- Overview Page ----------
with pages[0]:
    st.subheader("Snapshot")
    cols = st.columns(5)
    cols[0].metric("Patients", f"{n_patients:,}")
    cols[1].metric("Procedures (rows)", f"{n_procedures:,}")
    cols[2].metric("Unique ICD Procedure Codes", f"{n_codes:,}")
    if not np.isnan(median_age):
        cols[3].metric("Median Anchor Age", f"{median_age:.1f}")
    else:
        cols[3].metric("Median Anchor Age", "—")
    if isinstance(admissions_enriched, pd.DataFrame):
        cols[4].metric("Admissions", f"{n_admissions:,}")
    else:
        cols[4].metric("Admissions", "—")

    if not gender_counts.empty:
        fig_g = px.bar(gender_counts, x="gender", y="count", title="Gender Distribution")
        fig_g.update_layout(margin=dict(l=20, r=20, t=35, b=20))
        st.plotly_chart(fig_g, use_container_width=True)

    if "anchor_age" in patients_enriched.columns:
        cc1, cc2 = st.columns(2)
        fig_age = px.histogram(patients_enriched, x="anchor_age", nbins=40, title="Age Distribution (Anchor Age)")
        fig_age.update_layout(margin=dict(l=20, r=20, t=35, b=20))
        cc1.plotly_chart(fig_age, use_container_width=True)

        fig_pp = px.histogram(patients_enriched, x="procedures_per_patient", nbins=40, title="Procedures per Patient")
        fig_pp.update_layout(margin=dict(l=20, r=20, t=35, b=20))
        cc2.plotly_chart(fig_pp, use_container_width=True)

    if isinstance(admissions_enriched, pd.DataFrame) and admissions_enriched["los_days"].notna().any():
        fig_los = px.histogram(admissions_enriched, x="los_days", nbins=50, title="Length of Stay (days)")
        fig_los.update_layout(margin=dict(l=20, r=20, t=35, b=20))
        st.plotly_chart(fig_los, use_container_width=True)

    st.info("Interpretation: Cohort composition, intensity of procedural care, and (if available) hospital length of stay provide context for downstream analyses.")

# ---------- Procedures Page ----------
with pages[1]:
    st.subheader("Procedural Mix and Trends")

    # Filters
    c1, c2, c3 = st.columns(3)
    years_available = sorted([y for y in proc_merged["year"].dropna().unique()]) if "year" in proc_merged.columns else []
    year_sel = c1.multiselect("Filter by Year", years_available, default=years_available[:])

    icd_ver_available = sorted(proc_merged["icd_version"].dropna().unique().tolist())
    icd_ver_sel = c2.multiselect("ICD Version", icd_ver_available, default=icd_ver_available[:])

    top_n = c3.slider("Top N procedures", min_value=5, max_value=50, value=20, step=5)

    data_filt = proc_merged.copy()
    if year_sel:
        data_filt = data_filt[data_filt["year"].isin(year_sel)]
    if icd_ver_sel:
        data_filt = data_filt[data_filt["icd_version"].isin(icd_ver_sel)]

    # Top procedures
    top_proc = (
        data_filt["procedure_title"]
        .fillna(data_filt["icd_code"])
        .value_counts()
        .head(top_n)
        .rename_axis("procedure_title")
        .reset_index(name="count")
    )
    fig_top = px.bar(top_proc, x="count", y="procedure_title", orientation="h", title=f"Top {top_n} Procedures")
    fig_top.update_layout(yaxis=dict(automargin=True), margin=dict(l=20, r=20, t=35, b=20))
    st.plotly_chart(fig_top, use_container_width=True)

    # Trend over time (monthly)
    if "month" in data_filt.columns and data_filt["month"].notna().any():
        monthly = data_filt.groupby("month").size().reset_index(name="procedures")
        monthly = monthly.sort_values("month")
        fig_trend = px.line(monthly, x="month", y="procedures", markers=True, title="Procedures Over Time (Monthly)")
        fig_trend.update_layout(margin=dict(l=20, r=20, t=35, b=20), xaxis_tickangle=-45)
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("No valid chartdate/month data to show time trends.")

    st.info("Interpretation: The bar chart highlights the most frequent procedures, signaling common care pathways. The line trend (if available) surfaces seasonality or shifts in coding intensity.")

# ---------- Patient Profiles Page ----------
with pages[2]:
    st.subheader("Patient Profiles and Procedure Intensity")

    # Filters
    left, right = st.columns([1, 3])
    genders = sorted(patients_enriched["gender"].dropna().unique().tolist()) if "gender" in patients_enriched.columns else []
    gender_sel = left.multiselect("Gender", genders, default=genders[:])

    if "anchor_age" in patients_enriched.columns and patients_enriched["anchor_age"].notna().any():
        min_age = int(np.floor(patients_enriched["anchor_age"].min()))
        max_age = int(np.ceil(patients_enriched["anchor_age"].max()))
        age_range = left.slider("Age range", min_value=min_age, max_value=max_age, value=(min_age, max_age))
    else:
        age_range = (0, 100)

    dfp = patients_enriched.copy()
    if gender_sel and "gender" in dfp.columns:
        dfp = dfp[dfp["gender"].isin(gender_sel)]
    if "anchor_age" in dfp.columns and dfp["anchor_age"].notna().any():
        dfp = dfp[(dfp["anchor_age"] >= age_range[0]) & (dfp["anchor_age"] <= age_range[1])]

    if "anchor_age" in dfp.columns and dfp.shape[0] > 0:
        fig_sc = px.scatter(
            dfp,
            x="anchor_age",
            y="procedures_per_patient",
            color="gender" if "gender" in dfp.columns else None,
            title="Procedures per Patient vs. Age",
            trendline="ols" if len(dfp) > 10 else None,
        )
        fig_sc.update_layout(margin=dict(l=20, r=20, t=35, b=20))
        right.plotly_chart(fig_sc, use_container_width=True)

    # Table: sample patients with most procedures
    st.markdown("**Patients with highest procedure counts (sample)**")
    top_pat_tbl = dfp.sort_values("procedures_per_patient", ascending=False).head(25)
    cols_to_show = [c for c in ["subject_id", "gender", "anchor_age", "procedures_per_patient", "anchor_year_group"] if c in top_pat_tbl.columns]
    st.dataframe(top_pat_tbl[cols_to_show], use_container_width=True)

    st.info("Interpretation: See whether procedure intensity clusters by age or gender. The trendline indicates directionality only—not causation.")

# ---------- Qualitative Themes Page ----------
with pages[3]:
    st.subheader("Qualitative Signal from Procedure Titles")

    # Token frequency from long titles
    freq_df = tokenize_titles(proc_merged["procedure_title"])
    fig_tokens = px.bar(freq_df, x="token", y="count", title="Common Terms in Procedure Titles")
    fig_tokens.update_layout(margin=dict(l=20, r=20, t=35, b=20), xaxis_tickangle=-45)
    st.plotly_chart(fig_tokens, use_container_width=True)

    st.caption("Note: Simple token frequency (not lemmatized/stemmed). Consider clinical curation for domain stopwords (e.g., 'artery', 'insertion', 'repair').")
    st.info("Interpretation: Common tokens surface dominant modalities (e.g., central lines, cath, endoscopy). Use this to craft hypotheses about typical care pathways, then validate quantitatively.")

# ---------- Admissions & Outcomes Page ----------
admissions_tab_index = 4 if isinstance(admissions_enriched, pd.DataFrame) else None
if admissions_tab_index is not None:
    with pages[admissions_tab_index]:
        st.subheader("Admissions & Outcomes")

        if admissions_enriched["los_days"].notna().any():
            c1, c2 = st.columns(2)
            fig_los = px.histogram(admissions_enriched, x="los_days", nbins=60, title="Length of Stay (days)")
            fig_los.update_layout(margin=dict(l=20, r=20, t=35, b=20))
            c1.plotly_chart(fig_los, use_container_width=True)

            # Procedures per admission vs LOS
            tmp = admissions_enriched.dropna(subset=["los_days"])
            fig_sc2 = px.scatter(
                tmp, x="los_days", y="procedures_per_admission",
                title="Procedures per Admission vs LOS",
                trendline="ols" if len(tmp) > 10 else None
            )
            fig_sc2.update_layout(margin=dict(l=20, r=20, t=35, b=20))
            c2.plotly_chart(fig_sc2, use_container_width=True)

        # Discharge disposition distribution
        if "discharge_location" in admissions_enriched.columns:
            disp = admissions_enriched["discharge_location"].fillna("Unknown").value_counts().reset_index()
            disp.columns = ["discharge_location", "count"]
            disp_top = disp.head(20)
            fig_disp = px.bar(disp_top, x="count", y="discharge_location", orientation="h", title="Top Discharge Dispositions")
            fig_disp.update_layout(yaxis=dict(automargin=True), margin=dict(l=20, r=20, t=35, b=20))
            st.plotly_chart(fig_disp, use_container_width=True)

        # Mortality
        if "died_in_hosp" in admissions_enriched.columns:
            mort = (admissions_enriched["died_in_hosp"] == True).mean()
            st.metric("In-hospital Mortality (unadjusted)", f"{mort*100:.2f}%")

            # Mortality by age group (join on patients to get age)
            if "anchor_age" in patients_enriched.columns:
                # merge admissions with patient age
                adm_pat = admissions_enriched.merge(patients_enriched[["subject_id", "anchor_age"]], on="subject_id", how="left")
                adm_pat["age_group"] = assign_age_group(adm_pat["anchor_age"])
                mort_by_age = adm_pat.groupby("age_group")["died_in_hosp"].mean().reset_index()
                mort_by_age["rate"] = mort_by_age["died_in_hosp"] * 100.0
                fig_mort_age = px.bar(mort_by_age, x="age_group", y="rate", title="Mortality Rate by Age Group (%)")
                fig_mort_age.update_layout(margin=dict(l=20, r=20, t=35, b=20))
                st.plotly_chart(fig_mort_age, use_container_width=True)

        # LOS-normalized procedure rate by disposition (box)
        if "proc_per_day" in admissions_enriched.columns and "discharge_location" in admissions_enriched.columns:
            tmp2 = admissions_enriched.dropna(subset=["proc_per_day"]).copy()
            # Keep top 8 dispositions for legibility
            top_disp_cats = tmp2["discharge_location"].value_counts().index[:8].tolist()
            tmp2 = tmp2[tmp2["discharge_location"].isin(top_disp_cats)]
            fig_box = px.box(tmp2, x="discharge_location", y="proc_per_day", title="Procedures per LOS Day by Discharge Disposition")
            fig_box.update_layout(xaxis_tickangle=-30, margin=dict(l=20, r=20, t=35, b=20))
            st.plotly_chart(fig_box, use_container_width=True)

        st.info("Interpretation: LOS contextualizes resource utilization; procedures per day approximate care intensity per unit time. Discharge disposition and unadjusted mortality summarize downstream outcomes but do not control for case-mix.")

# ---------- Findings & Risks Page ----------
with pages[-1]:
    st.subheader("Key Findings, Interpretation, and Risks")

    # Example auto-filled stats
    med_pp = float(patients_enriched["procedures_per_patient"].median())
    p90_pp = float(patients_enriched["procedures_per_patient"].quantile(0.9))
    top_example = proc_merged["procedure_title"].fillna(proc_merged["icd_code"]).value_counts().index[:3].tolist()

    lines = []
    lines.append(f"- Median procedures per patient: **{med_pp:.0f}** (90th percentile: **{p90_pp:.0f}**).")
    if not np.isnan(median_age):
        lines.append(f"- Age distribution centers around **{median_age:.1f} years** (if reported).")
    if len(top_example) > 0:
        lines.append(f"- Most frequent procedures include: **{', '.join([str(t) for t in top_example])}**.")
    if isinstance(admissions_enriched, pd.DataFrame) and admissions_enriched["los_days"].notna().any():
        lines.append(f"- Median LOS: **{np.nanmedian(admissions_enriched['los_days']):.2f} days**; "
                     f"unadjusted mortality: **{(admissions_enriched['died_in_hosp']==True).mean()*100:.2f}%**.")

    st.markdown("**Findings (auto-generated summary):**\n" + "\n".join(lines))

    st.markdown(
        "**How to interpret:**\n"
        "- High procedure counts profile **care intensity** but not necessarily quality or outcomes.\n"
        "- LOS-normalized rates (procedures/day) help compare utilization across different stay lengths.\n"
        "- Discharge disposition and mortality are **unadjusted** snapshots; interpret with caution.\n"
    )

    caveats = [
        "- **Confounding & selection**: Case severity, service line, and admission type are not controlled here.",
        "- **Coding drift**: Changes in coding practices or ICD version mix can mimic or mask true clinical trends.",
        "- **Data quality**: Null/ambiguous dates affect trends; verify ETL. ",
        "- **Linkage assumptions**: Mortality is derived from admissions `deathtime`; confirm definition matches your study protocol."
    ]
    if not isinstance(admissions_enriched, pd.DataFrame):
        caveats.insert(0, "- **Missing admissions data**: LOS and outcomes limited until `admissions.csv` is provided.")

    st.markdown("**Risks & caveats:**\n" + "\n".join(caveats))

# ---------- Footer ----------
st.caption("© 2025 Procedures + Admissions dashboard. Educational use only; not for clinical decision-making.")
