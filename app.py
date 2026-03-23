import streamlit as st
import json
import os
import io
from openai import OpenAI

def extract_text_from_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            if not text.strip(): return None, "pdf_empty"
            return text, None
        except Exception as e: return None, str(e)
    elif name.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(uploaded_file.read()))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip()), None
        except Exception as e: return None, str(e)
    return None, "unsupported"

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
- certainty must be one of: clear, unclear, not_stated
  * clear = explicitly stated; unclear = ambiguous; not_stated = not mentioned
- mdt_agreement: agree, disagree, unclear, not_stated
- Infer medication route/schedule if not stated: Mirtazapine=oral/nocte, Lorazepam=oral/PRN, Zopiclone=oral/nocte PRN, Olanzapine=oral, depot injections=IM depot
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

TRIBUNAL_INPATIENT_PROMPT = """You are preparing a Responsible Clinician's report for an inpatient Mental Health Tribunal.
The patient is detained in hospital and is appealing against their detention under Section 2 or Section 3.

You have two sources of information. Use BOTH:

STRUCTURED EXTRACTED DATA:
{extracted_data}

FULL CLINICAL NOTES (raw text — use this for narrative detail, risk history, MSE findings, and background):
{raw_notes}

The raw notes contain important clinical detail that must be used, especially for risk history, circumstances of admission, mental state, and recommendations.

Return a JSON object with exactly these fields. Write in formal clinical prose. After substantive claims include EVIDENCE: with specific document/date citations.

{{
  "patient_name": "First name of patient",
  "rc_name": "Not documented — clinician to complete",
  "tribunal_type": "Inpatient detention appeal",

  "q3_factors_affecting_hearing": "Does the patient have any intellectual disability, physical disability, sensory impairment, or communication difficulty that would affect their ability to participate in a tribunal hearing? This is NOT about mental state symptoms. If no such factors are documented write: There are no known intellectual disabilities, physical disabilities, or communication difficulties that would affect the patient's ability to participate in a tribunal hearing.",

  "q4_adjustments": "Are any physical or communication adjustments needed for fair proceedings — such as interpreters, hearing loops, or accessible formats? If none needed write: No specific adjustments are required at this time.",

  "q5_forensic_history": "State any index offences or formal forensic history. If none, write this clearly. Then add a brief contextual note about any risk-relevant behaviours documented in the notes even if not resulting in formal proceedings — e.g. confrontation risk, threatening behaviour.",

  "q6_previous_mh_involvement": "List all previous mental health involvement including: any previous psychiatric admissions with dates; any previous community mental health input; any history of mental health difficulties managed in primary care. Draw from the full narrative notes, not just the structured data. If this is a first admission state this clearly.",

  "q7_reasons_previous_admissions": "Give reasons for any previous admissions or recalls. If this is the first admission write: This is the patient's first psychiatric admission.",

  "q8_circumstances_current_admission": "Write a detailed multi-paragraph clinical narrative covering: (1) relevant background history and any precipitating factors; (2) the prodromal period — how symptoms developed over time; (3) the specific symptoms and behaviours that raised concern, including the full risk picture — suicidal ideation, threats, dangerous behaviour, substance use; (4) what specifically precipitated the MHA assessment; (5) the legal basis for detention. This section must include the complete risk history that led to admission. Do not omit risk events.",

  "q9_mental_disorder_present": "Answer "Yes." or "No." only. Do not include diagnosis or symptoms here — those belong in section 10.",

  "q10_diagnosis": "State the diagnosis. Explain the clinical basis — what symptoms and history support it. Address any diagnostic uncertainty or differential diagnoses. In substance-related cases, address whether this is substance-induced or a primary disorder and what evidence supports that distinction.",

  "q11_learning_disability": "Does the patient have a learning disability? State yes or no with brief explanation.",

  "q12_detention_required": "Address NATURE and DEGREE as two explicitly separate concepts: NATURE — describe the type of mental disorder, its characteristic features, its pattern and course over time, associated features such as poor insight or non-adherence, and what typically happens when untreated. DEGREE — describe the current severity right now: specific symptoms present, behavioural disturbance and functional impairment, how this episode compares to baseline, and why the current degree of illness makes inpatient treatment appropriate. Then address: the patient's current insight and attitude to treatment; specific risk behaviours making community management unsafe; and the explicit causal chain from mental disorder through impaired insight to risk behaviour to need for detention.",

  "q13_treatment": "PHARMACOLOGICAL: all medications with doses and routes, compliance, any refused with reasons, whether IM medication has been required. NON-PHARMACOLOGICAL: psychology, OT, substance misuse referrals — state if offered, engaged with, or declined. COMMUNITY TEAM INVOLVEMENT: only include if a referral to EIT, CMHT, or other community team is documented — state outcome including if declined or not taken on; do not invent referrals. ENGAGEMENT: overall engagement and attitude. PLANNED: planned treatment going forward.",

  "q14_strengths": "List all strengths and positive factors including: engagement with staff; compliance with any treatment; emerging insight; social support; protective factors; any positive prognostic indicators. Draw from the full narrative notes.",

  "q15_current_progress": "Structure as: PRESENTATION AT ADMISSION — mental state, behaviour, and risk on the day of admission. PROGRESS DURING ADMISSION — how clinical picture evolved, key turning points. CURRENT NURSING OBSERVATIONS — most recent observations on behaviour, engagement, sleep, appetite, mood. INSIGHT — current level: does the patient understand they are unwell, do they accept treatment, has this changed. CAPACITY — one brief sentence only: whether the patient has been assessed as having capacity. Do not elaborate on legal implications here — that is for section 17.",

  "q16_medication_compliance": "List every medication with name, dose, route, and schedule — infer route and schedule from drug name if not stated (Mirtazapine is oral nocte, Lorazepam PRN is oral). State compliance with each. Explicitly state whether IM medication has been required — if not: 'No IM medication has been required during this admission.' Address medications offered but declined and reasons. Comment on likely future willingness. Note if refusal appears capacitous but influenced by ongoing symptoms.",

  "q17_mca_consideration": "Legal framework reasoning only — do not repeat the full capacity assessment, just reference it briefly. State whether the patient has or lacks capacity (from section 15). Then address: if the patient HAS capacity, state that the MHA remains the appropriate framework as the patient meets criteria for detention regardless of capacity — the MCA is not appropriate where MHA criteria are met. If the patient LACKS capacity, address whether a DoLS under the MCA 2005 would be appropriate and less restrictive.",

  "q18_incidents_self_harm_others": "This is a critical section. Document ALL incidents of harm, threats, or dangerous behaviour from BOTH the admission period and the period prior to admission. Draw from the full narrative notes. Include: suicidal ideation; threats of self-harm; actual self-harm; threatening behaviour towards others; any weapons involvement; risk-driven behaviour such as keeping objects for protection. Do not leave this section incomplete.",

  "q19_property_damage": "Document any property damage or threats. If none, state this clearly. Note any escalating behaviours that stopped short of actual damage but indicate risk.",

  "q20_section2_detention_justified": "This is a critical section. Provide detailed reasoning for why detention remains justified covering: (1) HEALTH — why detention is necessary for the patient's health; (2) SAFETY — why detention is necessary for the patient's safety, with reference to specific risk events; (3) PROTECTION OF OTHERS — any risk to third parties; (4) WHY COMMUNITY IS INSUFFICIENT — why the patient cannot be safely managed outside hospital at this time. The reasoning must be specific and evidence-based, not generic.",

  "q21_treatment_in_hospital_justified": "Address why inpatient treatment specifically is justified. This applies in all cases including Section 2. Cover: why hospital-based monitoring is needed; what risks are contained by the inpatient setting; why community treatment is not currently sufficient; what the inpatient setting provides that cannot be replicated in the community.",

  "q22_risk_if_discharged": "Draw from the pre-computed risk assessment. Structure as: RISK TO SELF; RISK TO OTHERS; RISK FROM OTHERS AND RETALIATION; RISK OF SELF-NEGLECT; RISK FROM SUBSTANCE USE; OVERALL SUMMARY. Use calibrated tribunal phrasing — for example 'moderate risk of deterioration and recurrence of harmful behaviours' not just 'moderate risk'. For mid-admission patients who are settled, state that risk is currently reduced in the structured inpatient setting but there remains a moderate risk of deterioration and recurrence if discharged prematurely.",

  "q23_community_risk_management": "Address how risks could or could not be managed in the community. Cover: whether any community treatment options exist; why they are currently insufficient — address engagement, insight, enforceability; what would need to change before community management would be safe.",

  "q24_recommendations": "Structure as: (1) CURRENT CLINICAL PICTURE — acknowledge any improvement, describe current mental state and trajectory. (2) CRITERIA FOR DETENTION — explicitly state whether criteria continue to be met and why. (3) CONSEQUENCES OF DISCHARGE — answer the tribunal's core question directly: if this patient were discharged now, what would likely happen? Be specific. Address: likely deterioration in mental state; risks to own health, safety, and physical wellbeing; risks to others; risk of treatment disengagement; substance use resumption if relevant; vulnerability in the community; likelihood of readmission. Use language such as 'discharge at this stage would be likely to result in...'. (4) WHAT NEEDS TO HAPPEN — what clinical progress is required before discharge would be appropriate.",

  "confidence_note": "List sections where information was limited and clinician completion is most important."
}}

Critical rules:
- Use BOTH the structured data AND the raw notes — the raw notes contain essential narrative detail
- Section 18 must include ALL documented risk events — this is non-negotiable
- Section 6 must include primary care history if documented in the raw notes
- Sections 3 and 4 are about physical/cognitive accessibility only, not mental state
- Never leave a field empty — provide content or state explicitly what is not documented
- Do not invent details not present in either source
- Use formal clinical language appropriate for a statutory tribunal report"""



TRIBUNAL_CTO_PROMPT = """You are preparing a Responsible Clinician's report for a Community Treatment Order (CTO) Mental Health Tribunal.
The patient is in the community under a CTO and is appealing against the CTO conditions.

You have two sources of information. Use BOTH:

STRUCTURED EXTRACTED DATA:
{extracted_data}

FULL CLINICAL NOTES (raw text — use this for narrative detail, risk history, MSE findings, and background):
{raw_notes}

Return a JSON object with exactly these fields. Write in formal clinical prose. After substantive claims include EVIDENCE: with specific document/date citations.

{{
  "patient_name": "First name of patient",
  "rc_name": "Not documented — clinician to complete",
  "tribunal_type": "CTO appeal",

  "q2_capacity_hearing": "Does the patient have capacity to decide whether to attend the tribunal and be represented? State clearly with date of assessment if documented.",

  "q3_factors_affecting_hearing": "Does the patient have any intellectual disability, physical disability, sensory impairment, or communication difficulty that would affect tribunal participation? This is NOT about mental state. If none write: There are no known intellectual disabilities, physical disabilities, or communication difficulties that would affect the patient's ability to participate in a tribunal hearing.",

  "q4_adjustments": "Are any physical or communication adjustments needed? If none write: No specific adjustments are required at this time.",

  "q5_forensic_history": "State any formal forensic history. If none, state this clearly. Add contextual note about any risk-relevant behaviours from the notes.",

  "q6_previous_mh_involvement": "All previous mental health involvement with dates — admissions, discharges, community input, primary care history. Draw from the full narrative notes.",

  "q7_reasons_previous_admissions": "Reasons for any previous admissions. If first admission state this.",

  "q8_circumstances_current_admission": "Detailed multi-paragraph narrative covering: background history; prodromal period; specific symptoms and behaviours raising concern; full risk picture; what led to MHA assessment; legal basis for admission and subsequent CTO.",

  "q9_mental_disorder": "State diagnosis and describe current symptoms. Address whether the disorder is of a nature and degree requiring ongoing CTO treatment.",

  "q10_learning_disability": "Does the patient have a learning disability? State yes or no.",

  "q11_treatment_appropriate": "Is mental disorder of a nature or degree such that ongoing CTO treatment is appropriate? Explain why community treatment under the CTO remains necessary.",

  "q12_treatment_details": "Cover: (1) PHARMACOLOGICAL — medications, doses, compliance; (2) NON-PHARMACOLOGICAL — community input, psychology, social work; (3) ENGAGEMENT — patient's engagement with CTO conditions; (4) PLANNED — future treatment intentions.",

  "q13_strengths": "All strengths and positive factors including engagement, compliance, support network, insight, and protective factors.",

  "q14_current_progress": "Current progress in the community covering behaviour, mental state, compliance with CTO, insight, and overall trajectory.",

  "q15_medication_compliance": "Patient's understanding of medication, compliance with depot or oral medication under the CTO, and likely future willingness if CTO were rescinded.",

  "q16_mca_consideration": "State capacity assessment findings. Explain why MHA rather than MCA is appropriate.",

  "q17_incidents": "ALL incidents of harm, threats, or dangerous behaviour from both the admission period and community — draw from full narrative notes. Do not omit risk events.",

  "q18_treatment_necessary": "Detailed reasoning for why ongoing CTO treatment is necessary covering: health necessity; safety necessity; risk to others if any; why voluntary community treatment without CTO would be insufficient.",

  "q19_risk_if_cto_rescinded": "Structured risk assessment if CTO were rescinded: RISK TO SELF; RISK TO OTHERS; RISK OF SELF-NEGLECT; RISK FROM NON-ADHERENCE TO MEDICATION; OVERALL SUMMARY with specific causal reasoning.",

  "q20_community_risk_management": "How risks are currently managed under the CTO. What would happen without compulsory powers — address enforceability, engagement, and medication adherence.",

  "q21_recommendations": "Full recommendation with reasoning covering: diagnosis; mental state; risk; insight; treatment compliance; why CTO criteria remain met; consequences of rescission.",

  "confidence_note": "Sections where information was limited and clinician completion is most important."
}}

Critical rules:
- Use BOTH structured data AND raw notes — raw notes contain essential narrative
- Section 17 must include ALL documented risk events
- Sections 3 and 4 are about physical/cognitive accessibility only
- Never leave a field empty
- Do not invent details not present in either source
- Use formal clinical language appropriate for a statutory tribunal report"""

import streamlit as st
import json
import os
import io
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
- certainty must be one of: clear, unclear, not_stated
  * clear = explicitly stated; unclear = ambiguous; not_stated = not mentioned
- mdt_agreement: agree, disagree, unclear, not_stated
- Infer medication route/schedule if not stated: Mirtazapine=oral/nocte, Lorazepam=oral/PRN, Zopiclone=oral/nocte PRN, Olanzapine=oral, depot injections=IM depot
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

TRIBUNAL_INPATIENT_PROMPT = """You are preparing a Responsible Clinician's report for an inpatient Mental Health Tribunal.
The patient is detained in hospital and is appealing against their detention under Section 2 or Section 3.

You have two sources of information. Use BOTH:

STRUCTURED EXTRACTED DATA:
{extracted_data}

FULL CLINICAL NOTES (raw text — use this for narrative detail, risk history, MSE findings, and background):
{raw_notes}

The raw notes contain important clinical detail that must be used, especially for risk history, circumstances of admission, mental state, and recommendations.

Return a JSON object with exactly these fields. Write in formal clinical prose. After substantive claims include EVIDENCE: with specific document/date citations.

{{
  "patient_name": "First name of patient",
  "rc_name": "Not documented — clinician to complete",
  "tribunal_type": "Inpatient detention appeal",

  "q3_factors_affecting_hearing": "Does the patient have any intellectual disability, physical disability, sensory impairment, or communication difficulty that would affect their ability to participate in a tribunal hearing? This is NOT about mental state symptoms. If no such factors are documented write: There are no known intellectual disabilities, physical disabilities, or communication difficulties that would affect the patient's ability to participate in a tribunal hearing.",

  "q4_adjustments": "Are any physical or communication adjustments needed for fair proceedings — such as interpreters, hearing loops, or accessible formats? If none needed write: No specific adjustments are required at this time.",

  "q5_forensic_history": "State any index offences or formal forensic history. If none, write this clearly. Then add a brief contextual note about any risk-relevant behaviours documented in the notes even if not resulting in formal proceedings — e.g. confrontation risk, threatening behaviour.",

  "q6_previous_mh_involvement": "List all previous mental health involvement including: any previous psychiatric admissions with dates; any previous community mental health input; any history of mental health difficulties managed in primary care. Draw from the full narrative notes, not just the structured data. If this is a first admission state this clearly.",

  "q7_reasons_previous_admissions": "Give reasons for any previous admissions or recalls. If this is the first admission write: This is the patient's first psychiatric admission.",

  "q8_circumstances_current_admission": "Write a detailed multi-paragraph clinical narrative covering: (1) relevant background history and any precipitating factors; (2) the prodromal period — how symptoms developed over time; (3) the specific symptoms and behaviours that raised concern, including the full risk picture — suicidal ideation, threats, dangerous behaviour, substance use; (4) what specifically precipitated the MHA assessment; (5) the legal basis for detention. This section must include the complete risk history that led to admission. Do not omit risk events.",

  "q9_mental_disorder_present": "State clearly that the patient is suffering from a mental disorder. Describe the specific symptoms that characterise the disorder — include thought content, perceptual disturbances, affect, behaviour, and insight. This should read as a clinical description not a checkbox.",

  "q10_diagnosis": "State the diagnosis. Explain the clinical basis — what symptoms and history support it. Address any diagnostic uncertainty or differential diagnoses. In substance-related cases, address whether this is substance-induced or a primary disorder and what evidence supports that distinction.",

  "q11_learning_disability": "Does the patient have a learning disability? State yes or no with brief explanation.",

  "q12_detention_required": "Address whether mental disorder requires detention for assessment or treatment. This is a critical section. Cover: (1) the nature and severity of the mental disorder; (2) the patient's insight and attitude to treatment; (3) the specific risk behaviours that make community management unsafe; (4) why hospital is necessary — what can be provided here that cannot be provided elsewhere. The causal chain should be explicit: mental disorder → impaired insight → risk behaviour → need for detention.",

  "q13_treatment": "Cover all of the following: (1) PHARMACOLOGICAL TREATMENT — all medications prescribed, doses, compliance, and any medications offered but declined with reasons; (2) NON-PHARMACOLOGICAL TREATMENT — psychology, OT, substance misuse referrals, or other interventions; (3) ENGAGEMENT — the patient's overall engagement with the treatment programme and attitude towards treatment; (4) PLANNED TREATMENT — what is planned going forward.",

  "q14_strengths": "List all strengths and positive factors including: engagement with staff; compliance with any treatment; emerging insight; social support; protective factors; any positive prognostic indicators. Draw from the full narrative notes.",

  "q15_current_progress": "Provide a detailed account covering: (1) PROGRESS — changes in mental state since admission; (2) BEHAVIOUR — ward behaviour, engagement, any incidents; (3) CAPACITY — capacity assessment findings with date if documented; (4) INSIGHT — current level of insight, understanding of illness, attitude to treatment; (5) OVERALL TRAJECTORY — is the patient improving, stable, or deteriorating.",

  "q16_medication_compliance": "Describe the patient's understanding of their medication, compliance with prescribed treatment, and likely future willingness to accept treatment. Address both pharmacological and non-pharmacological treatment separately.",

  "q17_mca_consideration": "Address the following: (1) Has the patient been assessed as having capacity to make treatment decisions? State the finding clearly with date if documented. (2) If the patient has capacity, explain why the MHA rather than the MCA is the appropriate framework. (3) If the patient lacks capacity, address whether a DoLS under MCA 2005 would be appropriate and less restrictive.",

  "q18_incidents_self_harm_others": "This is a critical section. Document ALL incidents of harm, threats, or dangerous behaviour from BOTH the admission period and the period prior to admission. Draw from the full narrative notes. Include: suicidal ideation; threats of self-harm; actual self-harm; threatening behaviour towards others; any weapons involvement; risk-driven behaviour such as keeping objects for protection. Do not leave this section incomplete.",

  "q19_property_damage": "Document any property damage or threats. If none, state this clearly. Note any escalating behaviours that stopped short of actual damage but indicate risk.",

  "q20_section2_detention_justified": "This is a critical section. Provide detailed reasoning for why detention remains justified covering: (1) HEALTH — why detention is necessary for the patient's health; (2) SAFETY — why detention is necessary for the patient's safety, with reference to specific risk events; (3) PROTECTION OF OTHERS — any risk to third parties; (4) WHY COMMUNITY IS INSUFFICIENT — why the patient cannot be safely managed outside hospital at this time. The reasoning must be specific and evidence-based, not generic.",

  "q21_treatment_in_hospital_justified": "Address why inpatient treatment specifically is justified. This applies in all cases including Section 2. Cover: why hospital-based monitoring is needed; what risks are contained by the inpatient setting; why community treatment is not currently sufficient; what the inpatient setting provides that cannot be replicated in the community.",

  "q22_risk_if_discharged": "Provide a structured risk assessment with the following headings: RISK TO SELF — include all documented self-harm ideation, threats, and behaviour with causal reasoning about what would happen if discharged; RISK TO OTHERS — document any risk to third parties; RISK OF SELF-NEGLECT AND VULNERABILITY — document any history of poor self-care or functional decline; RISK RELATED TO SUBSTANCE USE — if relevant, describe the relationship between substance use, mental state, and risk; OVERALL RISK SUMMARY — current risk level and key factors that would precipitate deterioration on discharge.",

  "q23_community_risk_management": "Address how risks could or could not be managed in the community. Cover: whether any community treatment options exist; why they are currently insufficient — address engagement, insight, enforceability; what would need to change before community management would be safe.",

  "q24_recommendations": "Provide a full recommendation with clinical reasoning covering: the current diagnosis and mental state; the risk picture; the patient's insight and attitude to treatment; the trajectory of improvement; and why the criteria for detention remain met at this time. The recommendation should read as a considered clinical opinion, not a formulaic statement.",

  "confidence_note": "List sections where information was limited and clinician completion is most important."
}}

Critical rules:
- Use BOTH the structured data AND the raw notes — the raw notes contain essential narrative detail
- Section 18 must include ALL documented risk events — this is non-negotiable
- Section 6 must include primary care history if documented in the raw notes
- Sections 3 and 4 are about physical/cognitive accessibility only, not mental state
- Never leave a field empty — provide content or state explicitly what is not documented
- Do not invent details not present in either source
- Use formal clinical language appropriate for a statutory tribunal report"""



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

  "q3_factors_affecting_hearing": "Are there any factors that may affect the patient's understanding or ability to cope with a hearing? This refers specifically to intellectual disabilities, physical disabilities, communication difficulties, or sensory impairments — NOT to mental state symptoms. State yes or no and explain. If no such factors are documented write: There are no known intellectual disabilities, physical disabilities, or communication difficulties that would affect the patient's ability to participate in a tribunal hearing.",

  "q4_adjustments": "Are there any adjustments the tribunal may consider for fair and just proceedings? This refers to physical or communication adjustments such as hearing loops, interpreters, or accessible formats — NOT mental state accommodations. State yes or no. If none required write: No adjustments are considered necessary.",

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


def compute_risk(extracted, notes, stage, api_key):
    client = OpenAI(api_key=api_key)
    raw_text = "\n\n---\n\n".join(f"NOTE {i+1}:\n{n}" for i,n in enumerate(notes))
    from string import Formatter
    prompt = RISK_ENGINE_PROMPT.format(
        extracted_data=json.dumps(extracted, indent=2, ensure_ascii=False),
        raw_notes=raw_text, admission_stage=stage)
    response = client.chat.completions.create(model="gpt-4o", temperature=0.0,
        messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}])
    output = response.choices[0].message.content.strip()
    if output.startswith("```"): output = "\n".join(l for l in output.splitlines() if not l.strip().startswith("```"))
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


def generate_tribunal(extracted_records, raw_notes, tribunal_type, api_key):
    client = OpenAI(api_key=api_key)
    raw_notes_text = "\n\n---\n\n".join(
        f"NOTE {i+1}:\n{note}" for i, note in enumerate(raw_notes)
    )
    if tribunal_type == "Inpatient detention appeal":
        prompt = TRIBUNAL_INPATIENT_PROMPT.format(
            extracted_data=json.dumps(extracted_records, indent=2, ensure_ascii=False),
            raw_notes=raw_notes_text
        )
    else:
        prompt = TRIBUNAL_CTO_PROMPT.format(
            extracted_data=json.dumps(extracted_records, indent=2, ensure_ascii=False),
            raw_notes=raw_notes_text
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

tab1, tab2, tab3 = st.tabs(["📄 Single note extraction", "📋 Section 3 recommendation", "⚖️ Tribunal report"])

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
                        tr_result = generate_tribunal(st.session_state.tr_extracted, st.session_state.tr_notes, tribunal_type_key, api_key)
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
