# app.py
import streamlit as st
from classification_helper import classify_pfas_molecule
from utils.utils import draw_mol, find_similar
import pandas as pd

st.set_page_config(page_title="PFAS Classifier", page_icon="🧪", layout="wide")

st.title("🧪 PFAS Classifier")
st.caption("Enter a SMILES string to check if it’s a PFAS and, if yes, see its classes.")

Property_List = [
    'LogP_pred', 'MP_pred', 'LogWS_pred', 'LogVP_pred',
    'LogHL_pred', 'LogKOA_pred', 'pKa_a_pred'
]
properties = ['LogP_pred', 'LogVP_pred']


# --- Sidebar Form ---
with st.form("pfas_form", clear_on_submit=False):
    smiles = st.text_input("SMILES", placeholder="e.g., O=C(O)C(F)(F)F")
    submitted = st.form_submit_button("Classify")

# --- Handle submit: run once, store in session state ---
if submitted:
    if not smiles.strip():
        st.error("Please enter a valid SMILES string.")
    else:
        try:
            st.session_state["smiles"] = smiles
            st.session_state["classes"] = classify_pfas_molecule(smiles)
            st.session_state["alternative_df"] = find_similar(smiles)
        except Exception as e:
            st.exception(e)


# --- Display results only if available ---
if "classes" in st.session_state:
    smiles = st.session_state["smiles"]
    classes = st.session_state["classes"]
    alternative_df = st.session_state["alternative_df"]

    # Show input molecule
    fig = draw_mol(smiles, padding=100)
    cols = st.columns(2)
    cols[0].pyplot(fig)

    primary, secondary = classes[0], classes[1]

    if "Not" not in primary:
        cols[1].success("✅ This molecule is identified as a PFAS.")

        with cols[1]:
            subcols = st.columns(2)
            subcols[0].markdown('Primary Class')
            subcols[0].subheader(primary)

            subcols[1].markdown('Secondary Class')
            subcols[1].subheader(secondary)

        # --- Alternatives Section ---
        cols = st.columns(3)
        num_alt = cols[0].slider('Select Number of Structurally Similar PFASs', 2, 10, 5)

        # Sliders for property ranges
        ranges = alternative_df[properties].agg(["min", "max"]).T
        selected_ranges = {}
        for prop, subcol in zip(properties, cols[1:]):
            min_val = float(ranges.loc[prop, "min"])
            max_val = float(ranges.loc[prop, "max"])
            selected = subcol.slider(
                    f"{prop} range",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                )
            selected_ranges[prop] = selected

        # --- Apply filters ---
        filtered_alternatives = alternative_df.copy()
        for prop, (low, high) in selected_ranges.items():
            filtered_alternatives = filtered_alternatives[
                (filtered_alternatives[prop] >= low) & (filtered_alternatives[prop] <= high)
            ]

        filtered_alternatives = filtered_alternatives.iloc[:num_alt]

        # --- Display alternatives ---
        for count, (ind, row) in enumerate(filtered_alternatives.iterrows(), start=1):
            st.subheader(f'Structurally Similar PFAS {count}')

            col1, col2 = st.columns([1.5, 3])
            fig = draw_mol(row['smiles'])
            col1.pyplot(fig)

            col2.markdown('Chemical Name')
            col2.subheader(row['Substance_Name'])

            subcols = col2.columns(3)
            subcols[0].markdown('Structure Formula')
            subcols[0].subheader(row['Structure_Formula'])

            subcols[1].markdown('Primary Class')
            subcols[1].subheader(row['First_Class'])

            subcols[2].markdown('Secondary Class')
            subcols[2].subheader(row['Second_Class'])

            subcols = col2.columns(2)
            subcols[0].markdown('Octanol-Water Partition Coefficient')
            subcols[0].subheader(row[properties[0]])

            subcols[1].markdown('Vapor Pressure')
            subcols[1].subheader(row[properties[1]])

            st.markdown('---')
    else:
        cols[1].info("❎ This molecule is not identified as a PFAS.")


with st.expander("About"):
    st.markdown("""
    - **What counts as PFAS?** Determined by the `classify_pfas_molecule` function.  
    - **Output:** A list of classes, e.g. `['PFAAs', 'PFCAs']`.  
    """)
