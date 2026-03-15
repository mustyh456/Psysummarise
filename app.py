import streamlit as st
import json
import os
from openai import OpenAI

st.set_page_config(page_title="PsySummarise", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.mono { font-family: 'IBM Plex Mono', monospace; }
.stTextArea textarea { font-family: 'IBM Plex Mono', monospace; font-size: 13px; }
.stButton > button {
    background: #3b82f6; color: white; border: none; border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace; font-weight: 500;
    padding: 0.6rem 2rem; width: 100%; font-size: 14px;
}
.stButton > button:hover { background: #2563eb; }
</style>
""", unsafe_allow_html=True)

SCHEMA_FILE = "extraction_schema.json"
SYSTEM_PROMPT = "You output only valid JSON. Do not include explanations, markdown, or code fences."

EXTRACTION_PROMPT = """You are a clinical NLP system that extracts structured information from psychiatric ward round notes.

Extract ALL relevant clinical events and return a single JSON object that strictly matches this schema:
{schema}

Rules:
- patient_id: use the patient's first name as it appears in the note, or "Unknown" if not present
- doc_id: use the document type if identifiable (e.g. "WR1", "Discharge"), otherwise "Unknown"
- For any field you cannot determine from the note, use null
- adherence must be one of: good, partial, poor, unknown
- certainty must be one of: clear, unclear
- action_date and event_date must be YYYY-MM-DD format, or null
- Include evidence_quote for every item where certainty is "clear"
- icd10: provide the ICD-10 code where you can confidently infer it, otherwise null
- event_type rules:
  * "admission" = patient being admitted for the first time in this episode
  * "section_change" = legal status noted or changed during ongoing admission
  * "discharge" = patient being discharged or discharge being planned
  * In ward round notes, "Legal Status: Section X MHA" almost always means section_change

Ward round note:
\"\"\"{note}\"\"\"

Return only the JSON object."""

S3_SYSTEM = """You are an expert psychiatrist helping to draft a Section 3 Mental Health Act recommendation.
You write clearly, clinically, and concisely. You never fabricate clinical details.
You only use information explicitly present in the notes provided.
You output only valid JSON."""

S3_PROMPT = """Based on the following extracted clinical data from multiple psychiatric documents,
draft a Section 3 MHA recommendation (Form A8) for medical review.

Extracted clinical data:
{extracted_data}

Return a JSON object with exactly these fields:
{{
  "patient_name": "First name of patient",

  "prior_acquaintance": "State whether you had previous acquaintance with the patient before examination. If unclear from notes write: Not documented — clinician to complete.",

  "nature_of_disorder": "Describe the diagnosed mental disorder using the statutory phrase 'mental disorder of a nature and degree which makes it appropriate for the patient to receive medical treatment in a hospital'. Follow with the specific diagnosis, how long symptoms have been present, and the clinical picture. After your prose, add a line break and write EVIDENCE: followed by bullet points citing specific documents and dates from the extracted data that support this, e.g. '- Admission 01/01: [observation]'.",

  "current_symptoms": "Describe the patient's current symptoms and behaviour in clinical terms. Draw on MSE findings, nursing observations, and ward round entries. Be specific — include thought content, perception, affect, behaviour, and insight. After your prose add EVIDENCE: with bullet points citing specific document entries and dates.",

  "risk_to_self": "Describe risk to the patient's own health and safety including self-neglect, medication non-compliance, impaired judgement, and any history of self-harm or suicidal ideation. After your prose add EVIDENCE: with bullet points citing specific documents and dates. If genuinely not documented write: No evidence of risk to self documented in available notes — clinician to review.",

  "risk_to_others": "Describe any risk to other persons documented in the notes with specific evidence. If not documented write: No evidence of risk to others documented in available notes — clinician to review.",

  "why_informal_insufficient": "Explain why informal admission is not appropriate. Address the patient's insight into their condition, their capacity or willingness to consent to voluntary treatment, and whether they would likely disengage if not detained. Cite specific evidence from the notes.",

  "why_community_insufficient": "Explain why community treatment is not appropriate. Reference the acuity of presentation, need for close monitoring, structured medication management, and MDT input that cannot be safely provided in the community. Cite specific evidence.",

  "ongoing_treatment_needed": "Explain why continued inpatient treatment under Section 3 is required. Focus on: the patient's mental health not yet being optimised; the need for ongoing MDT assessment and medication monitoring; the importance of developing insight and capacity; and the requirement for safe and planned discharge. Do not say 'treatment cannot be completed within 28 days'. Cite specific evidence from the notes.",

  "medication_history": "Summarise medication history including previous medications, reasons for any changes or discontinuation, current medication, compliance, and response. Cite specific documents and dates.",

  "recommendation": "A concluding statement using language such as: 'I recommend detention under Section 3 of the Mental Health Act as further inpatient treatment is required for optimisation of [patient]'s mental health, to ensure appropriate clinical management, and to allow safe and planned discharge when clinically indicated.' Tailor to the specific clinical picture.",

  "confidence_note": "List any areas where the available notes provided limited information and clinician review is especially important. Be specific about which fields need the most attention."
}}

Critical rules:
- Use the statutory phrase 'mental disorder of a nature and degree' in the nature_of_disorder field
- Always include EVIDENCE citations after each clinical claim — this is essential for clinician trust and auditability
- Where information is genuinely absent write clearly: 'Not documented in available notes — clinician to complete'
- Never invent or infer clinical details not present in the notes
- Use clear, formal clinical language appropriate for a statutory MHA document
- The tone should reflect a senior clinician making a considered recommendation, not a bureaucratic checklist
- Risk fields should never be left blank — either cite evidence or explicitly state it is not documented"""




def load_schema():
    if os.path.exists(SCHEMA_FILE):
        with open(SCHEMA_FILE) as f:
            return json.load(f)
    return {}


def extract_note(note_text, api_key):
    client = OpenAI(api_key=api_key)
    schema = load_schema()
    prompt = EXTRACTION_PROMPT.format(
        schema=json.dumps(schema, indent=2),
        note=note_text
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    output = response.choices[0].message.content.strip()
    if output.startswith("```"):
        output = "\n".join(l for l in output.splitlines() if not l.strip().startswith("```"))
    return json.loads(output)


def generate_s3(extracted_records, api_key):
    client = OpenAI(api_key=api_key)
    prompt = S3_PROMPT.format(
        extracted_data=json.dumps(extracted_records, indent=2, ensure_ascii=False)
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        messages=[
            {"role": "system", "content": S3_SYSTEM},
            {"role": "user", "content": prompt}
        ]
    )
    output = response.choices[0].message.content.strip()
    if output.startswith("```"):
        output = "\n".join(l for l in output.splitlines() if not l.strip().startswith("```"))
    return json.loads(output)


def render_clinical_summary(data):
    patient = data.get("patient_id") or "Unknown"
    doc = data.get("doc_id") or "Unknown"

    st.markdown(f"### {patient} — {doc}")

    # Admissions
    admissions = data.get("admissions", [])
    if admissions:
        st.markdown("#### Admissions & Legal Status")
        for adm in admissions:
            with st.container(border=True):
                cols = st.columns([2, 2, 2])
                cols[0].metric("Legal status", adm.get("legal_status") or "—")
                cols[1].metric("Event type", adm.get("event_type") or "—")
                cols[2].metric("Date", adm.get("event_date") or "Unknown")
                if adm.get("reason"):
                    st.markdown(f"**Reason:** {adm['reason']}")
                if adm.get("destination"):
                    st.markdown(f"**Destination:** {adm['destination']}")
                if adm.get("notes"):
                    st.markdown(f"**Notes:** {adm['notes']}")
                if adm.get("evidence_quote"):
                    st.caption(f'"{adm["evidence_quote"]}"')

    # Diagnoses
    diagnoses = data.get("diagnoses", [])
    if diagnoses:
        st.markdown("#### Diagnoses")
        for dx in diagnoses:
            with st.container(border=True):
                label = dx.get("dx_label") or "Unknown"
                st.markdown(f"**{label}**")
                cols = st.columns(4)
                cols[0].markdown(f"ICD-10: `{dx.get('icd10') or '—'}`")
                cols[1].markdown(f"Status: `{dx.get('status') or '—'}`")
                cols[2].markdown(f"MDT: `{dx.get('mdt_agreement') or '—'}`")
                cols[3].markdown(f"Certainty: `{dx.get('certainty') or '—'}`")
                if dx.get("evidence_quote"):
                    st.caption(f'"{dx["evidence_quote"]}"')

    # Medications
    medications = data.get("medications", [])
    if medications:
        st.markdown("#### Medications")
        action_icons = {"start": "🟢", "continue": "🔵", "change": "🟡", "stop": "🔴"}
        for med in medications:
            with st.container(border=True):
                action = med.get("action") or "?"
                icon = action_icons.get(action, "⚪")
                name = med.get("med_name") or "Unknown"
                dose = med.get("dose_text") or ""
                st.markdown(f"{icon} **{action.upper()}** — {name} {dose}")
                cols = st.columns(4)
                cols[0].markdown(f"Route: `{med.get('route') or '—'}`")
                cols[1].markdown(f"Schedule: `{med.get('schedule') or '—'}`")
                cols[2].markdown(f"Adherence: `{med.get('adherence') or '—'}`")
                cols[3].markdown(f"Certainty: `{med.get('certainty') or '—'}`")
                if med.get("response"):
                    st.markdown(f"**Response:** {med['response']}")
                if med.get("side_effects"):
                    st.markdown(f"**Side effects:** {', '.join(med['side_effects'])}")
                if med.get("reason_change"):
                    st.markdown(f"**Reason for change:** {med['reason_change']}")
                if med.get("evidence_quote"):
                    st.caption(f'"{med["evidence_quote"]}"')

    if not admissions and not diagnoses and not medications:
        st.info("No clinical data extracted from this note.")


def render_s3(s3_data, patient_name):
    st.warning("⚠️ AI-assisted draft only. All content must be reviewed and approved by the responsible clinician before use.")

    st.markdown("""
> *I am approved under section 12 of the Act as having special experience in the diagnosis or treatment of mental disorder.*
""")

    fields = [
        ("Prior acquaintance with patient", "prior_acquaintance"),
        ("Nature of mental disorder", "nature_of_disorder"),
        ("Current symptoms and mental state", "current_symptoms"),
        ("Risk to self", "risk_to_self"),
        ("Risk to others", "risk_to_others"),
        ("Why informal admission is insufficient", "why_informal_insufficient"),
        ("Why community treatment is insufficient", "why_community_insufficient"),
        ("Why ongoing inpatient treatment is required", "ongoing_treatment_needed"),
        ("Medication history and compliance", "medication_history"),
        ("Recommendation", "recommendation"),
    ]

    edits = {}
    for label, key in fields:
        val = s3_data.get(key) or "Not documented in available notes"
        edits[key] = st.text_area(label, value=val, height=100, key=f"s3_{key}")

    confidence = s3_data.get("confidence_note")
    if confidence:
        st.info(f"**AI confidence note:** {confidence}")

    st.divider()
    plain = f"""MEDICAL RECOMMENDATION FOR ADMISSION FOR TREATMENT (SECTION 3 MHA 1983 — FORM A8)
Patient: {patient_name}

I am approved under section 12 of the Act as having special experience in the diagnosis or treatment of mental disorder.

PRIOR ACQUAINTANCE
{edits.get('prior_acquaintance', '')}

NATURE OF MENTAL DISORDER
{edits.get('nature_of_disorder', '')}

CURRENT SYMPTOMS AND MENTAL STATE
{edits.get('current_symptoms', '')}

RISK TO SELF
{edits.get('risk_to_self', '')}

RISK TO OTHERS
{edits.get('risk_to_others', '')}

WHY INFORMAL ADMISSION IS INSUFFICIENT
{edits.get('why_informal_insufficient', '')}

WHY COMMUNITY TREATMENT IS INSUFFICIENT
{edits.get('why_community_insufficient', '')}

WHY ONGOING INPATIENT TREATMENT IS REQUIRED
{edits.get('ongoing_treatment_needed', '')}

MEDICATION HISTORY AND COMPLIANCE
{edits.get('medication_history', '')}

RECOMMENDATION
{edits.get('recommendation', '')}

---
AI-ASSISTED DRAFT — Must be reviewed and signed by an approved clinician before submission.
Generated by PsySummarise (research prototype). Not validated for clinical use.
"""

    st.download_button(
        "Download Section 3 draft (.txt)",
        data=plain,
        file_name=f"Section3_{patient_name.replace(' ', '_')}.txt",
        mime="text/plain"
    )


# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("# 🧠 PsySummarise")
st.markdown("*Structured extraction from psychiatric documentation — research prototype*")
st.divider()

with st.sidebar:
    st.markdown("### Settings")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    st.divider()
    st.caption("PsySummarise extracts structured clinical information from psychiatric ward round notes using GPT-4o.\n\nNot validated for clinical use.")

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["Single note extraction", "Section 3 recommendation"])

# ── Tab 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    note_text = st.text_area(
        "Paste ward round note",
        height=280,
        placeholder="Paste a psychiatric ward round note, admission summary, or discharge letter here..."
    )
    if st.button("Extract structured data →", key="extract_single"):
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        elif not note_text.strip():
            st.warning("Please paste a ward round note above.")
        else:
            with st.spinner("Extracting..."):
                try:
                    result = extract_note(note_text, api_key)
                    st.success("Extraction complete.")
                    st.divider()
                    summary_tab, json_tab = st.tabs(["Clinical summary", "Raw JSON"])
                    with summary_tab:
                        render_clinical_summary(result)
                    with json_tab:
                        st.json(result)
                        st.download_button(
                            "Download JSON",
                            data=json.dumps(result, indent=2, ensure_ascii=False),
                            file_name=f"psysummarise_{result.get('patient_id', 'output')}.json",
                            mime="application/json"
                        )
                except json.JSONDecodeError as e:
                    st.error(f"Model returned invalid JSON: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Tab 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Build a Section 3 recommendation from multiple notes")
    st.caption("Add each ward round note or clinical document one at a time. The system extracts from each and synthesises a draft Section 3 recommendation across all notes.")

    if "s3_notes" not in st.session_state:
        st.session_state.s3_notes = []
    if "s3_extracted" not in st.session_state:
        st.session_state.s3_extracted = []

    new_note = st.text_area(
        "Paste a note",
        height=200,
        placeholder="Paste a ward round note, admission summary, MDT review, or any clinical document...",
        key="s3_input"
    )

    col_add, col_clear = st.columns([3, 1])
    with col_add:
        if st.button("Add note →", key="add_note"):
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            elif not new_note.strip():
                st.warning("Please paste a note above.")
            else:
                with st.spinner(f"Extracting note {len(st.session_state.s3_notes) + 1}..."):
                    try:
                        extracted = extract_note(new_note, api_key)
                        st.session_state.s3_notes.append(new_note)
                        st.session_state.s3_extracted.append(extracted)
                        st.success(f"Note {len(st.session_state.s3_notes)} added — {extracted.get('patient_id', '?')} / {extracted.get('doc_id', '?')}")
                    except Exception as e:
                        st.error(f"Extraction failed: {e}")

    with col_clear:
        if st.button("Clear all", key="clear_notes"):
            st.session_state.s3_notes = []
            st.session_state.s3_extracted = []
            st.rerun()

    if st.session_state.s3_extracted:
        st.markdown(f"**{len(st.session_state.s3_extracted)} note(s) added:**")
        for i, rec in enumerate(st.session_state.s3_extracted):
            n_meds = len(rec.get("medications", []))
            n_dx = len(rec.get("diagnoses", []))
            n_adm = len(rec.get("admissions", []))
            st.caption(f"{i+1}. {rec.get('patient_id','?')} / {rec.get('doc_id','?')} — {n_meds} medication(s), {n_dx} diagnosis(es), {n_adm} admission event(s)")

        st.divider()
        if st.button("Generate Section 3 recommendation →", key="gen_s3"):
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            else:
                with st.spinner("Synthesising Section 3 recommendation..."):
                    try:
                        s3_result = generate_s3(st.session_state.s3_extracted, api_key)
                        st.success("Draft generated. Review all fields carefully before use.")
                        st.divider()
                        patient_name = s3_result.get("patient_name") or \
                            (st.session_state.s3_extracted[0].get("patient_id", "Patient")
                             if st.session_state.s3_extracted else "Patient")
                        draft_tab, source_tab = st.tabs(["Draft recommendation", "Source data"])
                        with draft_tab:
                            render_s3(s3_result, patient_name)
                        with source_tab:
                            st.json(st.session_state.s3_extracted)
                    except json.JSONDecodeError as e:
                        st.error(f"Model returned invalid JSON: {e}")
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.info("Add at least one note above to begin.")
