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




TRIBUNAL_SYSTEM = """You are an expert consultant psychiatrist preparing a statutory Mental Health Tribunal report.
You write clearly, clinically, and with appropriate legal precision.
You only use information explicitly present in the notes provided.
You never fabricate clinical details.
You output only valid JSON."""

TRIBUNAL_INPATIENT_PROMPT = """Based on the following extracted clinical data from multiple psychiatric documents,
prepare a Responsible Clinician's report for an inpatient Mental Health Tribunal hearing.
The patient is detained in hospital and is appealing against their detention under Section 2 or Section 3.

Extracted clinical data:
{extracted_data}

Return a JSON object with exactly these fields. For each field, after your clinical prose write EVIDENCE: followed by bullet points citing specific document names and dates from the extracted data.

{{
  "patient_name": "First name of patient",
  "rc_name": "Not documented — clinician to complete",
  "tribunal_type": "Inpatient detention appeal",

  "q3_factors_affecting_hearing": "Are there any factors that may affect the patient's understanding or ability to cope with a hearing? State yes or no and explain.",

  "q4_adjustments": "Are there any adjustments the tribunal may consider for fair proceedings? State yes or no and explain.",

  "q5_forensic_history": "Details of any index offence(s) and other relevant forensic history. If none documented write: No forensic history documented in available notes.",

  "q6_previous_mh_involvement": "Dates of previous involvement with mental health services including admissions, discharges, and recalls. Include dates where documented.",

  "q7_reasons_previous_admissions": "Reasons for any previous admissions or recall to hospital. If none write: No previous admissions documented.",

  "q8_circumstances_current_admission": "Circumstances leading up to the current admission. Include presenting symptoms, behaviour, risk factors, and what led to assessment under the MHA. Cite specific evidence.",

  "q9_mental_disorder_present": "Is the patient now suffering from a mental disorder? Answer yes or no with clinical reasoning.",

  "q10_diagnosis": "Has a diagnosis been made and what is it? Provide the diagnosis and clinical basis for it.",

  "q11_learning_disability": "Does the patient have a learning disability? State yes or no.",

  "q12_detention_required": "Is there any mental disorder present which requires the patient to be detained for assessment and/or medical treatment? Answer yes or no with reasoning.",

  "q13_treatment": "What appropriate and available medical treatment has been prescribed, provided, offered or is planned? Include pharmacological and non-pharmacological interventions.",

  "q14_strengths": "What are the strengths or positive factors relating to the patient? Include engagement, support network, insight, and any protective factors.",

  "q15_current_progress": "Summary of current progress, behaviour, capacity and insight. Include MSE findings, behavioural observations, and any changes since admission.",

  "q16_medication_compliance": "What is the patient's understanding of, compliance with, and likely future willingness to accept prescribed medication or treatment?",

  "q17_mca_consideration": "In the case of an eligible compliant patient lacking capacity — whether deprivation of liberty under the MCA 2005 would be appropriate and less restrictive.",

  "q18_incidents_self_harm_others": "Details of any incidents where the patient has harmed themselves or others, or threatened harm. Cover both during admission and prior to admission.",

  "q19_property_damage": "Details of any incidents of property damage or threats to damage property. If none write: No incidents of property damage documented.",

  "q20_section2_detention_justified": "In Section 2 cases: is detention justified or necessary for health, safety, or protection of others? Provide detailed clinical reasoning covering health, safety, and risk to others separately.",

  "q21_treatment_in_hospital_justified": "In all other cases: is inpatient treatment justified or necessary for health, safety, or protection of others?",

  "q22_risk_if_discharged": "If the patient were discharged, would they likely act dangerously to themselves or others? Explain how risks could be managed in the community.",

  "q23_community_risk_management": "How can any risks be managed effectively in the community, including the use of any lawful conditions or recall powers?",

  "q24_recommendations": "Recommendations to the tribunal with full clinical reasoning. Be specific about why criteria remain met and what continued detention allows.",

  "confidence_note": "List any sections where the available notes provided limited information and clinician review is especially important."
}}

Critical rules:
- Include EVIDENCE citations after every substantive clinical claim
- Never leave a field blank — either provide clinical content or state what is not documented
- Use formal language appropriate for a statutory tribunal report
- Where information is absent write: Not documented in available notes — clinician to complete
- Do not invent or infer details not present in the notes"""


TRIBUNAL_CTO_PROMPT = """Based on the following extracted clinical data from multiple psychiatric documents,
prepare a Responsible Clinician's report for a Community Treatment Order (CTO) Mental Health Tribunal hearing.
The patient is in the community under a CTO and is appealing against the CTO conditions.

Extracted clinical data:
{extracted_data}

Return a JSON object with exactly these fields. For each field, after your clinical prose write EVIDENCE: followed by bullet points citing specific document names and dates from the extracted data.

{{
  "patient_name": "First name of patient",
  "rc_name": "Not documented — clinician to complete",
  "tribunal_type": "CTO appeal",

  "q2_capacity_hearing": "Does the patient have capacity to decide whether to attend the tribunal and be represented? State opinion clearly.",

  "q3_factors_affecting_hearing": "Are there any factors that may affect the patient's understanding or ability to cope with a hearing? State yes or no and explain.",

  "q4_adjustments": "Are there any adjustments the tribunal may consider for fair proceedings? State yes or no and explain.",

  "q5_forensic_history": "Details of any index offence(s) and other relevant forensic history. If none documented write: No forensic history documented in available notes.",

  "q6_previous_mh_involvement": "Dates of previous involvement with mental health services including admissions, discharges, and recalls.",

  "q7_reasons_previous_admissions": "Reasons for any previous admissions or recall to hospital.",

  "q8_circumstances_current_admission": "Circumstances leading up to the most recent admission. Include presenting symptoms, behaviour, risk factors, and what led to MHA assessment. Cite specific evidence.",

  "q9_mental_disorder": "Is the patient suffering from a mental disorder within the meaning of the MHA? Provide diagnosis and clinical basis.",

  "q10_learning_disability": "Does the patient have a learning disability? State yes or no.",

  "q11_treatment_appropriate": "Is mental disorder of a nature or degree such that medical treatment is appropriate? Explain why ongoing CTO treatment remains necessary.",

  "q12_treatment_details": "Details of treatment prescribed, provided, offered or planned under the CTO. Include depot medication, community follow-up, and any psychological input.",

  "q13_strengths": "Strengths or positive factors relating to the patient. Include engagement, support network, insight, and protective factors.",

  "q14_current_progress": "Summary of current progress, behaviour, capacity and insight in the community.",

  "q15_medication_compliance": "Patient's understanding of, compliance with, and likely future willingness to accept prescribed medication under the CTO.",

  "q16_mca_consideration": "Whether deprivation of liberty under the MCA 2005 would be appropriate and less restrictive than the CTO.",

  "q17_incidents": "Details of any incidents of self-harm, harm to others, threats, or property damage. Cover both during admission and in the community.",

  "q18_treatment_necessary": "Is it necessary for the patient's health or safety, or for the protection of others, that they receive medical treatment under the CTO? Provide detailed reasoning.",

  "q19_risk_if_cto_rescinded": "If the CTO were rescinded, would the patient likely act dangerously to themselves or others? Be specific about relapse risk and medication non-adherence.",

  "q20_community_risk_management": "How can risks be managed in the community with or without the CTO? Address what would happen without compulsory powers.",

  "q21_recommendations": "Recommendations to the tribunal with full reasoning. Address why the CTO criteria remain met and the consequences of rescission.",

  "confidence_note": "List any sections where the available notes provided limited information and clinician review is especially important."
}}

Critical rules:
- Include EVIDENCE citations after every substantive clinical claim
- Frame all reasoning around CTO criteria, not inpatient detention criteria
- Never leave a field blank — either provide content or state what is not documented
- Use formal language appropriate for a statutory tribunal report
- Where information is absent write: Not documented in available notes — clinician to complete
- Do not invent or infer details not present in the notes"""


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


def generate_tribunal(extracted_records, tribunal_type, api_key):
    client = OpenAI(api_key=api_key)
    if tribunal_type == "Inpatient detention appeal":
        prompt = TRIBUNAL_INPATIENT_PROMPT.format(
            extracted_data=json.dumps(extracted_records, indent=2, ensure_ascii=False)
        )
    else:
        prompt = TRIBUNAL_CTO_PROMPT.format(
            extracted_data=json.dumps(extracted_records, indent=2, ensure_ascii=False)
        )
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        messages=[
            {"role": "system", "content": TRIBUNAL_SYSTEM},
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

# ── Tab 3: Tribunal report ────────────────────────────────────────────────────
with tab3:
    st.markdown("### Generate a tribunal report from multiple notes")
    st.caption("Add each clinical document one at a time. Select the tribunal type before generating.")

    if "tr_notes" not in st.session_state:
        st.session_state.tr_notes = []
    if "tr_extracted" not in st.session_state:
        st.session_state.tr_extracted = []

    tribunal_type = st.radio(
        "Tribunal type",
        ["Inpatient detention appeal (Section 2 / Section 3)", "CTO appeal"],
        horizontal=True
    )
    tribunal_type_key = "Inpatient detention appeal" if "Inpatient" in tribunal_type else "CTO appeal"

    tr_note = st.text_area(
        "Paste a note",
        height=200,
        placeholder="Paste a ward round note, admission summary, MDT review, or any clinical document...",
        key="tr_input"
    )

    tr_col1, tr_col2 = st.columns([3, 1])
    with tr_col1:
        tr_add = st.button("Add note →", key="tr_add_note")
    with tr_col2:
        if st.button("Clear all", key="tr_clear"):
            st.session_state.tr_notes = []
            st.session_state.tr_extracted = []
            st.rerun()

    if tr_add:
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        elif not tr_note.strip():
            st.warning("Please paste a note above.")
        else:
            with st.spinner(f"Extracting note {len(st.session_state.tr_notes)+1}..."):
                try:
                    extracted = extract_note(tr_note, api_key)
                    st.session_state.tr_notes.append(tr_note)
                    st.session_state.tr_extracted.append(extracted)
                    st.success(f"Note {len(st.session_state.tr_notes)} added — {extracted.get('patient_id','?')} / {extracted.get('doc_id','?')}")
                except Exception as e:
                    st.error(f"Extraction failed: {e}")

    if st.session_state.tr_extracted:
        st.markdown(f"**{len(st.session_state.tr_extracted)} note(s) loaded:**")
        for i, rec in enumerate(st.session_state.tr_extracted):
            n_meds = len(rec.get("medications", []))
            n_dx = len(rec.get("diagnoses", []))
            n_adm = len(rec.get("admissions", []))
            st.caption(f"{i+1}. {rec.get('patient_id','?')} / {rec.get('doc_id','?')} — {n_meds} medication(s), {n_dx} diagnosis(es), {n_adm} admission event(s)")

        st.divider()
        if st.button("Generate tribunal report →", key="gen_tribunal"):
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            else:
                with st.spinner("Generating tribunal report — this may take 30-60 seconds..."):
                    try:
                        tr_result = generate_tribunal(st.session_state.tr_extracted, tribunal_type_key, api_key)
                        st.success("Draft generated. Review all sections carefully before use.")
                        st.divider()

                        patient_name = tr_result.get("patient_name") or                             (st.session_state.tr_extracted[0].get("patient_id", "Patient")
                             if st.session_state.tr_extracted else "Patient")

                        st.warning("⚠️ AI-assisted draft. Must be reviewed and approved by the Responsible Clinician before submission to any tribunal. Not for direct clinical use.")

                        tr_draft_tab, tr_source_tab = st.tabs(["Draft report", "Source data"])

                        with tr_draft_tab:
                            st.markdown(f"**{tr_result.get('tribunal_type','Tribunal')} — {patient_name}**")
                            st.caption(f"RC: {tr_result.get('rc_name','Not documented — clinician to complete')}")
                            st.divider()

                            if tribunal_type_key == "Inpatient detention appeal":
                                sections = [
                                    ("3. Factors affecting hearing", "q3_factors_affecting_hearing"),
                                    ("4. Adjustments for fair proceedings", "q4_adjustments"),
                                    ("5. Forensic history", "q5_forensic_history"),
                                    ("6. Previous MH involvement", "q6_previous_mh_involvement"),
                                    ("7. Reasons for previous admissions", "q7_reasons_previous_admissions"),
                                    ("8. Circumstances of current admission", "q8_circumstances_current_admission"),
                                    ("9. Mental disorder present?", "q9_mental_disorder_present"),
                                    ("10. Diagnosis", "q10_diagnosis"),
                                    ("11. Learning disability?", "q11_learning_disability"),
                                    ("12. Detention required?", "q12_detention_required"),
                                    ("13. Treatment available", "q13_treatment"),
                                    ("14. Strengths and positive factors", "q14_strengths"),
                                    ("15. Current progress, behaviour, capacity and insight", "q15_current_progress"),
                                    ("16. Medication compliance and future willingness", "q16_medication_compliance"),
                                    ("17. MCA consideration", "q17_mca_consideration"),
                                    ("18. Incidents of harm or threats", "q18_incidents_self_harm_others"),
                                    ("19. Property damage", "q19_property_damage"),
                                    ("20. Section 2 — is detention justified?", "q20_section2_detention_justified"),
                                    ("21. Inpatient treatment justified?", "q21_treatment_in_hospital_justified"),
                                    ("22. Risk if discharged", "q22_risk_if_discharged"),
                                    ("23. Community risk management", "q23_community_risk_management"),
                                    ("24. Recommendations", "q24_recommendations"),
                                ]
                            else:
                                sections = [
                                    ("2. Capacity for hearing", "q2_capacity_hearing"),
                                    ("3. Factors affecting hearing", "q3_factors_affecting_hearing"),
                                    ("4. Adjustments for fair proceedings", "q4_adjustments"),
                                    ("5. Forensic history", "q5_forensic_history"),
                                    ("6. Previous MH involvement", "q6_previous_mh_involvement"),
                                    ("7. Reasons for previous admissions", "q7_reasons_previous_admissions"),
                                    ("8. Circumstances of current admission", "q8_circumstances_current_admission"),
                                    ("9. Mental disorder and diagnosis", "q9_mental_disorder"),
                                    ("10. Learning disability?", "q10_learning_disability"),
                                    ("11. CTO treatment appropriate?", "q11_treatment_appropriate"),
                                    ("12. Treatment details", "q12_treatment_details"),
                                    ("13. Strengths and positive factors", "q13_strengths"),
                                    ("14. Current progress in community", "q14_current_progress"),
                                    ("15. Medication compliance", "q15_medication_compliance"),
                                    ("16. MCA consideration", "q16_mca_consideration"),
                                    ("17. Incidents of harm or threats", "q17_incidents"),
                                    ("18. Treatment necessary under CTO?", "q18_treatment_necessary"),
                                    ("19. Risk if CTO rescinded", "q19_risk_if_cto_rescinded"),
                                    ("20. Community risk management", "q20_community_risk_management"),
                                    ("21. Recommendations", "q21_recommendations"),
                                ]

                            tr_edits = {}
                            for label, key in sections:
                                val = tr_result.get(key) or "Not documented in available notes — clinician to complete"
                                st.markdown(f"**{label}**")
                                tr_edits[key] = st.text_area(
                                    label=label,
                                    value=val,
                                    height=120,
                                    key=f"tr_{key}",
                                    label_visibility="collapsed"
                                )

                            confidence = tr_result.get("confidence_note")
                            if confidence:
                                with st.expander("AI confidence note — sections needing most clinician attention"):
                                    st.info(confidence)

                            st.divider()
                            plain = f"RESPONSIBLE CLINICIAN'S REPORT FOR MENTAL HEALTH TRIBUNAL\n"
                            plain += f"Tribunal type: {tr_result.get('tribunal_type','')}\n"
                            plain += f"Patient: {patient_name}\n"
                            plain += f"RC: {tr_result.get('rc_name','')}\n\n"
                            for label, key in sections:
                                plain += f"{label.upper()}\n{tr_edits.get(key,'')}\n\n"
                            plain += "---\nAI-ASSISTED DRAFT — Must be reviewed and signed by the Responsible Clinician before submission.\nGenerated by PsySummarise (research prototype). Not validated for clinical use.\n"

                            st.download_button(
                                "⬇️ Download tribunal report draft (.txt)",
                                data=plain,
                                file_name=f"TribunalReport_{patient_name.replace(' ','_')}.txt",
                                mime="text/plain"
                            )

                        with tr_source_tab:
                            st.json(st.session_state.tr_extracted)

                    except json.JSONDecodeError as e:
                        st.error(f"Model returned invalid JSON: {e}")
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.info("Add at least one note above to begin.")
