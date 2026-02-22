"""
PROJECT: Stratacyst Platinum v3.0
AUTHOR: ADITHYA VIKRAM
DATE: February 16 2026
INSTITUTION: REDMOND MIDDLE SCHOOL
PURPOSE: Multi-Modal AI for CFTR Variant Pathogenicity Classification & Biophysical Stabilization.
VERSION: Platinum-Stable (Audited)
"""
# LICENSE: MIT License
# Copyright (c) 2026 ADITHYA VIKRAM
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation, to deal in the Software without restriction, 
# including without limitation the rights to use, copy, modify, merge, publish, 
# distribute, sublicense, and/or sell copies of the Software.

from datetime import datetime
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.utils import resample

# --- 1. DATA ORCHESTRATOR ---
@st.cache_resource
def load_and_merge_data():
    """Loads and merges the clinical and biophysical datasets."""
    try:
        # Loading the three pillars of the dataset
        isef_df = pd.read_csv('FINAL_ISEF_COMPARISON_1176_VARIANTS.csv')
        stab_df = pd.read_csv('CLINICAL_DEVICE_V3_STABILIZED.csv')
        audit_df = pd.read_csv('MASTER_AUDIT_ENRICHED.csv')

        # Standardizing the target column
        isef_df = isef_df.rename(columns={'TRUE_CLASS': 'CLASS'})
        
        # Merging Clinical Audit data (Sweat, FEV1, PI, PSI)
        audit_subset = audit_df[['Variant', 'sweat_chloride', 'pi_percent', 'fev1_percent', 'ps_infection_rate']]
        master_df = pd.merge(isef_df, audit_subset, on='Variant', how='left')
        
        # Merging Biophysical targets (GFI, MS)
        stab_subset = stab_df[['Variant', 'STABILIZED_GFI', 'FINAL_PRED_MS']]
        master_df = pd.merge(master_df, stab_subset, on='Variant', how='left')
        
        # Data Integrity: Drop rows without a Class and fill clinical gaps with medians
        master_df = master_df.dropna(subset=['CLASS'])
        for col in ['sweat_chloride', 'pi_percent', 'fev1_percent', 'ps_infection_rate']:
            master_df[col] = master_df[col].fillna(master_df[col].median())
            
        return master_df
    except Exception as e:
        st.error(f"ENGINE CORE FAILURE during Data Load: {e}")
        return pd.DataFrame()

# --- 2. BIOPHYSICAL FEATURE EXTRACTOR ---
def stratacyst_extract_precision_features(v):
    """Converts a variant name into a numerical biophysical vector."""
    v = str(v).upper().strip().replace("[", "").replace("]", "")
    if ";" in v: v = v.split(";")[0]

    # Structural Logic
    is_protein_inframe = bool(re.search(r'^[A-Z]\d+DEL', v))
    is_frameshift = any(x in v for x in ['FS', 'X', 'STOP']) or ('DEL' in v and not is_protein_inframe)
    is_splice = any(x in v for x in ['+', '-', 'IVS', 'C.']) and not is_protein_inframe
    
    # Extract Protein Position
    pos_match = re.search(r'(\d+)', v)
    pos = int(pos_match.group(1)) if pos_match else 0
    
    # Domain Mapping
    nonsense = 1 if is_frameshift else 0
    nbd = 1 if (389 <= pos <= 673) or (1174 <= pos <= 1480) else 0
    msd = 1 if (1 <= pos <= 388) or (845 <= pos <= 1173) else 0
    
    # Functional Categories
    folding = 1 if is_protein_inframe or (msd and not nonsense) else 0
    gating = 1 if pos in [551, 549, 178, 1244, 1251, 1349, 1235] else 0
    cond = 1 if pos in [117, 1152, 334, 347, 206, 455, 1066, 67] else 0
    
    return [pos, nonsense, folding, (1 if is_splice else 0), gating, cond, nbd, msd]

def stratacyst_get_modulator_recommendation(p_class, p_ms, p_gfi):
    """Matches the predicted class to the appropriate modulator profile."""
    if p_class == 'I':
        return "⚠️ NO CURRENT MODULATOR", "Class I variants (Nonsense) typically require read-through agents or gene therapy. Standard modulators (Trikafta) may have limited efficacy unless a splice-site is present."
    
    if p_class == 'II':
        return "💊 TRIPLE COMBINATION (CORRECTORS)", "Requires Elexacaftor/Tezacaftor/Ivacaftor (Trikafta). The primary defect is folding (Low MS Score). Correctors are needed to chaperone the protein to the surface."
    
    if p_class == 'III':
        return "💊 POTENTIATOR (KALYDECO)", "Primarily requires Ivacaftor. The protein is at the surface (Stable MS) but the 'gate' is locked. Potentiators force the gate open."
    
    if p_class == 'IV':
        return "💊 POTENTIATOR / DUAL COMBO", "Ivacaftor or Symdeko. Conductance is limited. Potentiators will maximize the flow of ions through the existing channels."
    
    if p_class == 'V':
        return "💊 DUAL COMBINATION / MODULATORS", "The defect is reduced quantity. Standard modulators can help maximize the function of the limited protein that reaches the surface."
    
    return "UNKNOWN", "Further biophysical analysis required."

# --- 3. AUDIT & VISUALIZATION TOOLS ---
def display_stratacyst_logic_audit(v_name, bits, sw, fe, pi, ps, p_gfi, p_ms, p_class, conf):
    """Generates the deep-reasoning logic for the UI."""
    pos, nonsense, folding, splice, gating, cond, nbd, msd = bits
    gap = abs(sw - pi)

    st.markdown(f"### 🧠 Stratacyst Logic Audit: {v_name}")
    col1, col2 = st.columns(2)

    with col1:
        domain = "NBD" if nbd else "MSD" if msd else "Non-Coding/Splice"
        site_type = "Gating" if gating else "Conductance" if cond else "Folding/Structural"
        st.info(f"**Genomic Anchor:** Position **{pos}** ({domain} Domain)")
        st.write(f"Engine identification: **{site_type}** defect.")
        if nonsense: st.write("🔴 Nonsense/FS detected: High probability of truncated protein.")

    with col2:
        st.info(f"**Biophysical Pulse:** S-GFI: {p_gfi:.2f} | MS: {p_ms:.2f}")
        if p_ms < 15:
            st.write("🔴 **Stability Alert:** Low MS score suggests severe proteostasis failure.")
        elif p_gfi > 40:
            st.write("🟢 **Conductance Priority:** High S-GFI suggests partial channel function.")

    st.divider()
    if gap > 40:
        st.warning(f"**Phenotypic Gap ({gap:.1f}):** Sweat chloride and PI are discordant. Weight shifted to biophysics.")
    else:
        st.success(f"**Phenotypic Alignment:** Clinical markers consistent with Class {p_class}.")

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Stratacyst Platinum System")
st.sidebar.caption("Developed by Adithya Vikram, Redmond Middle School")
st.sidebar.caption("ISEF Project Category: Computational Biology")
st.sidebar.caption("Note that this system is a research prototype for educational purposes and should not be used for clinical decision-making. Always consult healthcare/medical professionals first.")

def display_stratacyst_analysis_plots(variant_data, clf_features, classifier):
    """Generates Local and Global importance bar charts."""
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Local Neural Audit")
        local_imp = pd.DataFrame({
            'Feature': clf_features,
            'Weight': classifier.feature_importances_ * variant_data[clf_features].values.flatten()
        }).sort_values(by='Weight', ascending=False)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Weight', y='Feature', data=local_imp, palette="flare", ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("🌍 Global Engine Priority")
        global_imp = pd.DataFrame({
            'Feature': clf_features,
            'Importance': classifier.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=global_imp, palette="mako", ax=ax)
        st.pyplot(fig)

# --- 4. TRAINING PIPELINE ---
@st.cache_resource
def train_platinum_engine():
    df = load_and_merge_data()
    if df.empty: return None

    # 1. Feature Prep
    bio_cols = ['pos', 'nonsense', 'folding', 'splice', 'gating', 'cond', 'nbd', 'msd']
    bio_data = df['Variant'].apply(lambda x: pd.Series(stratacyst_extract_precision_features(x)))
    bio_data.columns = bio_cols
    df = pd.concat([df, bio_data], axis=1)
    df['gap'] = abs(df['sweat_chloride'] - df['pi_percent'])
    
    # 2. Regression Stage
    reg_features = bio_cols + ['sweat_chloride', 'pi_percent', 'fev1_percent', 'ps_infection_rate']
    gfi_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    ms_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    
    gfi_model.fit(df.dropna(subset=['STABILIZED_GFI'])[reg_features], df.dropna(subset=['STABILIZED_GFI'])['STABILIZED_GFI'])
    ms_model.fit(df.dropna(subset=['FINAL_PRED_MS'])[reg_features], df.dropna(subset=['FINAL_PRED_MS'])['FINAL_PRED_MS'])
    
    # Impute missing biophysics using the regressor
    df.loc[df['STABILIZED_GFI'].isna(), 'STABILIZED_GFI'] = gfi_model.predict(df.loc[df['STABILIZED_GFI'].isna(), reg_features])
    df.loc[df['FINAL_PRED_MS'].isna(), 'FINAL_PRED_MS'] = ms_model.predict(df.loc[df['FINAL_PRED_MS'].isna(), reg_features])

    # 3. Balanced Classification Stage (Fixing the Memorization Trap)
    clf_features = ['nonsense', 'folding', 'splice', 'gating', 'cond', 'nbd', 'msd', 
                    'sweat_chloride', 'fev1_percent', 'STABILIZED_GFI', 'FINAL_PRED_MS', 'gap', 'ps_infection_rate']
    
    valid_classes = ['I', 'II', 'III', 'IV', 'V']
    df_filtered = df[df['CLASS'].isin(valid_classes)]
    
    # Reduced n_samples to 500 to prevent overfitting via 600% duplication
    balanced_list = [resample(df_filtered[df_filtered['CLASS'] == cls], 
                              replace=True, n_samples=500, random_state=42) for cls in valid_classes]
    df_balanced = pd.concat(balanced_list)
    
    classifier = RandomForestClassifier(n_estimators=500, criterion='entropy', class_weight='balanced', random_state=42)
    classifier.fit(df_balanced[clf_features], df_balanced['CLASS'].astype(str))
    
    return gfi_model, ms_model, classifier, clf_features

def show_stratacyst_history_log():
    """Displays the persistent session research log."""
    if 'history' in st.session_state and st.session_state.history:
        st.divider()
        st.subheader("📜 Stratacyst Research Log")
        history_df = pd.DataFrame(st.session_state.history)
        
        # Display the table
        st.dataframe(history_df, use_container_width=True)
        
        # CSV Export Logic
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Full Research Report",
            data=csv,
            file_name=f"Stratacyst_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
        
# --- 5. MAIN INTERFACE ---
def main_stratacyst_interface():
    st.set_page_config(page_title="Stratacyst Platinum", layout="wide")
    
    # Sidebar & Initialization
    st.sidebar.title("Stratacyst Control")
    with st.spinner("Synchronizing Engine..."):
        engine_data = train_platinum_engine()
        if not engine_data: st.stop()
        gfi_model, ms_model, classifier, clf_features = engine_data

    # Input Markers
    variant_name = st.sidebar.text_input("Variant Name", value="F508del")
    sw_cl = st.sidebar.number_input("Sweat Chloride", 0, 160, 100)
    fev1 = st.sidebar.number_input("FEV1 (%)", 10, 120, 45)
    pi = st.sidebar.number_input("Pancreatic Insufficiency (%)", 0, 100, 95)
    psi = st.sidebar.number_input("Pseudomonas Rate (%)", 0, 100, 80)

    if st.sidebar.button("🚀 RUN ANALYSIS"):
        bio_bits = stratacyst_extract_precision_features(variant_name)
        reg_input = np.array(bio_bits + [sw_cl, pi, fev1, psi]).reshape(1, -1)
        
        # Pred 1: Biophysics
        pred_gfi = gfi_model.predict(reg_input)[0]
        pred_ms = ms_model.predict(reg_input)[0]
        
        # Pred 2: Class
        gap = abs(sw_cl - pi)
        clf_input_data = bio_bits[1:] + [sw_cl, fev1, pred_gfi, pred_ms, gap, psi]
        clf_input = np.array(clf_input_data).reshape(1, -1)
        
        pred_class = classifier.predict(clf_input)[0]
        confidence = max(classifier.predict_proba(clf_input)[0]) * 100

        # Output Display
        st.title(f"🧬 Analysis: {variant_name}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Class", f"Class {pred_class}")
        m2.metric("S-GFI (Conductance)", f"{pred_gfi:.2f}")
        m3.metric("MS Score (Stability)", f"{pred_ms:.2f}")
        m4.metric("Confidence", f"{confidence:.1f}%")
        
        # --- TREATMENT RECOMMENDATION LAYER ---
        rec_title, rec_desc = stratacyst_get_modulator_recommendation(pred_class, pred_ms, pred_gfi)
        
        st.success(f"### 🛡️ Therapeutic Recommendation: {rec_title}")
        st.info(rec_desc)

        # Visuals & Logic Audit
        display_stratacyst_logic_audit(variant_name, bio_bits, sw_cl, fev1, pi, psi, pred_gfi, pred_ms, pred_class, confidence)
        variant_series = pd.Series(clf_input_data, index=clf_features)
        display_stratacyst_analysis_plots(variant_series, clf_features, classifier)
        
        # Update Research Log
        # --- SAVE TO HISTORY ---
        new_entry = {
            "Timestamp": datetime.now().strftime("%H:%M:%S"),
            "Variant": variant_name,
            
            "Class": pred_class,
            "S-GFI": round(pred_gfi, 2),
            "MS Score": round(pred_ms, 2),
            "Sweat Cl": sw_cl,
            "FEV1": fev1,
            "Confidence": f"{confidence:.1f}%",
            "pi": pi,
            "psi": psi,
            "Logic Notes": f"Gap: {gap:.1f} | BioBits: {bio_bits}",
            "Stratacyst Logic": f"Domain: {'NBD' if bio_bits[6] else 'MSD' if bio_bits[7] else 'Non-Coding/Splice'} | Site: {'Gating' if bio_bits[4] else 'Conductance' if bio_bits[5] else 'Folding/Structural'}"            
        }
        
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        # Add new result to the top of the list
        st.session_state.history.insert(0, new_entry)

if __name__ == "__main__":
    main_stratacyst_interface()

# Show the log at the bottom of the page
    show_stratacyst_history_log()