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

RISK_ENGINE_PROMPT = """You are a consultant psychiatrist performing a structured pre-tribunal risk assessment.

Analyse the following clinical data and produce a calibrated risk assessment across five domains.

EXTRACTED CLINICAL DATA:
{extracted_data}

RAW CLINICAL NOTES:
{raw_notes}

ADMISSION STAGE: {admission_stage}

Risk level definitions:
- LOW: no recent incidents, no active symptoms driving risk, adequate insight and engagement
- MODERATE: recent risk history OR ongoing symptoms OR partial insight, currently mitigated by structure/support
- HIGH: recent serious incident PLUS active symptoms AND poor insight/impulsivity — use sparingly

Stage guidance:
- Acute admission: risk levels reflect the presenting crisis
- Mid-admission settled: inpatient risk is reduced but community risk remains — do not rate as HIGH unless very recent serious incident
- Discharge-ready: focus on community risk

Return a JSON object:
{{
  "patient_name": "First name",
  "admission_stage": "{admission_stage}",
  "domains": {{
    "risk_to_self": {{
      "level": "LOW/MODERATE/HIGH",
      "past_factors": "Historical incidents and risk events",
      "present_factors": "Current risk in inpatient setting",
      "future_factors": "Likely risk if discharged prematurely",
      "increasing_factors": ["factors that increase risk"],
      "reducing_factors": ["factors that reduce risk"],
      "evidence": ["specific clinical events from the notes"],
      "tribunal_paragraph": "Write 2-3 sentences in natural clinical prose as used in real tribunal reports. Do not use EVIDENCE: labels. Embed facts naturally. Distinguish current inpatient risk from likely community risk. Use calibrated language: 'Risk is currently reduced within the structured inpatient setting; however, there remains a moderate risk of deterioration and recurrence of harmful behaviour if discharged prematurely...'"
    }},
    "risk_to_others": {{
      "level": "LOW/MODERATE/HIGH",
      "past_factors": "",
      "present_factors": "",
      "future_factors": "",
      "increasing_factors": [],
      "reducing_factors": [],
      "evidence": [],
      "tribunal_paragraph": "Natural clinical prose. Calibrated to stage. If no risk: state clearly."
    }},
    "risk_from_others": {{
      "level": "LOW/MODERATE/HIGH",
      "past_factors": "",
      "present_factors": "",
      "future_factors": "",
      "increasing_factors": [],
      "reducing_factors": [],
      "evidence": [],
      "tribunal_paragraph": "Address retaliation risk and vulnerability. Natural prose."
    }},
    "risk_of_self_neglect": {{
      "level": "LOW/MODERATE/HIGH",
      "past_factors": "",
      "present_factors": "",
      "future_factors": "",
      "increasing_factors": [],
      "reducing_factors": [],
      "evidence": [],
      "tribunal_paragraph": "Natural clinical prose."
    }},
    "risk_substance_use": {{
      "level": "LOW/MODERATE/HIGH",
      "past_factors": "",
      "present_factors": "",
      "future_factors": "",
      "increasing_factors": [],
      "reducing_factors": [],
      "evidence": [],
      "tribunal_paragraph": "Natural clinical prose."
    }}
  }},
  "overall_risk_summary": "2-3 sentences of natural clinical prose. State current inpatient risk level and likely community risk separately. Use calibrated language.",
  "stage_rationale": "One sentence explaining why risk levels are calibrated to this stage."
}}"""


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

  "nature_of_disorder": "Describe the diagnosed mental disorder using the statutory phrase 'mental disorder of a nature and degree which makes it appropriate for the patient to receive medical treatment in a hospital'. Follow with the specific diagnosis, how long symptoms have been present, and the clinical picture. ",

  "current_symptoms": "Describe the patient's current symptoms and behaviour in clinical terms. Draw on MSE findings, nursing observations, and ward round entries. Be specific — include thought content, perception, affect, behaviour, and insight. ",

  "risk_to_self": "Describe risk to the patient's own health and safety including self-neglect, medication non-compliance, impaired judgement, and any history of self-harm or suicidal ideation.  If genuinely not documented write: No evidence of risk to self documented in available notes — clinician to review.",

  "risk_to_others": "Describe any risk to other persons documented in the notes  If not documented write: No evidence of risk to others documented in available notes — clinician to review.",

  "why_informal_insufficient": "Explain why informal admission is not appropriate. Address the patient's insight into their condition, their capacity or willingness to consent to voluntary treatment, and whether they would likely disengage if not detained.",

  "why_community_insufficient": "Explain why community treatment is not appropriate. Reference the acuity of presentation, need for close monitoring, structured medication management, and MDT input that cannot be safely provided in the community.",

  "ongoing_treatment_needed": "Explain why continued inpatient treatment under Section 3 is required. Focus on: the patient's mental health not yet being optimised; the need for ongoing MDT assessment and medication monitoring; the importance of developing insight and capacity; and the requirement for safe and planned discharge. Do not say 'treatment cannot be completed within 28 days'.",

  "medication_history": "Summarise medication history including previous medications, reasons for any changes or discontinuation, current medication, compliance, and response.",

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

You have three sources of information. Use ALL of them:

STRUCTURED EXTRACTED DATA:
{extracted_data}

FULL CLINICAL NOTES (raw text — use this for narrative detail, risk history, MSE findings, and background):
{raw_notes}

PRE-COMPUTED RISK ASSESSMENT:
{risk_assessment}

ADMISSION STAGE: {admission_stage}

The raw notes contain important clinical detail that must be used, especially for risk history, circumstances of admission, mental state, and recommendations.

CRITICAL WRITING RULES — apply to every section:
- Write in natural NHS consultant prose. Do not use bullet points anywhere in the output. Do not use "EVIDENCE:" labels.
- Every section must use consequence-based, causal reasoning — not just description. Show WHY, not just WHAT.
- Use temporal language throughout. Show how the clinical picture has evolved over time.
- Sections 15, 20, 23, and 24 are the most important sections. Write each with a minimum of 3-4 sentences of substantive clinical reasoning.
- Do not write summaries. Write clinical reasoning and clinical judgment.
- Where the patient has improved, acknowledge this — but then address what risks remain and why discharge remains premature.
- The tribunal's core question is always: "Should this person remain detained?" Answer it directly in sections 20, 23, and 24.
- Never use generic phrases like "appropriate monitoring" or "intensive monitoring" without specifying what is being monitored and why.

Return a JSON object with exactly these fields. Write in formal clinical prose.

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

  "q9_mental_disorder_present": "Answer 'Yes.' Then add one brief sentence only, identifying the category of disorder (e.g. 'Yes. Jordan is suffering from a substance-induced psychotic disorder.'). Do not elaborate further — full clinical description belongs in sections 10 and 12.",

  "q10_diagnosis": "State the diagnosis only. One or two sentences maximum. Do not include explanatory paragraphs, clinical basis, or differential diagnoses here. Example format: 'Substance-induced psychosis (MDMA-related). A primary psychotic disorder cannot be excluded at this stage.' That is all that is required in this field.",

  "q11_learning_disability": "Does the patient have a learning disability? State yes or no with brief explanation.",

  "q12_detention_required": "Answer the statutory question directly: 'Is there a mental disorder of a nature or degree which warrants the patient's detention in hospital for assessment and/or medical treatment?' Answer Yes or No, then write one brief sentence of justification only — for example: 'Yes. Jordan is suffering from a mental disorder of a nature and degree that warrants detention for further assessment and treatment.' Do NOT include extended clinical reasoning here. The full nature and degree analysis belongs in section 21.",

  "q13_treatment": "Cover all of the following: (1) PHARMACOLOGICAL TREATMENT — all medications prescribed, doses, compliance, and any medications offered but declined with reasons; (2) NON-PHARMACOLOGICAL TREATMENT — psychology, OT, substance misuse referrals, or other interventions; (3) ENGAGEMENT — the patient's overall engagement with the treatment programme and attitude towards treatment; (4) PLANNED TREATMENT — what is planned going forward.",

  "q14_strengths": "List all strengths and positive factors including: engagement with staff; compliance with any treatment; emerging insight; social support; protective factors; any positive prognostic indicators. Draw from the full narrative notes.",

  "q15_current_progress": "Write four clearly delineated paragraphs. PRESENTATION AT ADMISSION: describe the mental state, specific risk behaviours, and presenting symptoms on the day of admission — include the specific events that led to admission. PROGRESS DURING ADMISSION: describe how the clinical picture has evolved since admission — what has improved, what has not changed, and what key clinical turning points occurred. Include specific events or changes where possible. CURRENT NURSING OBSERVATIONS: describe the most recent ward behaviour — engagement with staff, sleep, appetite, agitation, incidents or absence of incidents. INSIGHT: state clearly the current level — does the patient accept they are unwell, do they accept the need for treatment, do they understand the severity of their presentation, and has this changed since admission. CAPACITY: one brief sentence only — state the finding. Do not elaborate here; legal framework discussion belongs in section 17.",

  "q16_medication_compliance": "Describe the patient's understanding of their medication, compliance with prescribed treatment, and likely future willingness to accept treatment. Address both pharmacological and non-pharmacological treatment separately.",

  "q17_mca_consideration": "Address the following: (1) Has the patient been assessed as having capacity to make treatment decisions? State the finding clearly with date if documented. (2) If the patient has capacity, explain why the MHA rather than the MCA is the appropriate framework. (3) If the patient lacks capacity, address whether a DoLS under MCA 2005 would be appropriate and less restrictive.",

  "q18_incidents_self_harm_others": "This is a critical section. Document ALL incidents of harm, threats, or dangerous behaviour from BOTH the admission period and the period prior to admission. Draw from the full narrative notes. Include: suicidal ideation; threats of self-harm; actual self-harm; threatening behaviour towards others; any weapons involvement; risk-driven behaviour such as keeping objects for protection. Do not leave this section incomplete.",

  "q19_property_damage": "State only what is documented. If there is no evidence of property damage or threats to damage property, write exactly: 'There is no evidence of property damage or threats to damage property documented in the available notes.' Do NOT speculate about future risk of property damage or infer risk from other behaviours. This section must be factual only.",

  "q20_section2_detention_justified": "This is a critical section. Write four paragraphs of clinical reasoning — do not summarise. HEALTH: explain specifically why detention is necessary for the patient's health — address the nature and severity of the mental disorder, the treatment required, and what would happen to the patient's mental health without hospital treatment. SAFETY: explain why detention is necessary for the patient's safety — reference specific risk events by name (suicidal ideation, self-harm threats, dangerous behaviour), explain the causal chain from mental disorder to risk behaviour, and state why this risk cannot be safely managed in the community. PROTECTION OF OTHERS: address any risk to third parties — if risk exists, describe it specifically; if no risk, state this clearly. WHY COMMUNITY IS INSUFFICIENT: explain directly why the patient cannot be safely managed outside hospital right now — address insight, treatment adherence, absence of enforceable community powers, and what would likely happen if detained were lifted. Do not write generic statements — use the specific facts of this case.",

  "q21_treatment_in_hospital_justified": "This is a critical section. You MUST address NATURE and DEGREE as two explicitly defined concepts — this is a legal requirement. NATURE OF THE DISORDER: describe the type of mental disorder, its characteristic features, its typical pattern and course, and the features of this patient's disorder that make it dangerous when untreated — specifically impaired insight, tendency to disengage from treatment, and the risk behaviours associated with the disorder. DEGREE OF THE DISORDER: describe the current severity right now — specific active symptoms, current level of behavioural disturbance, current level of insight, functional impairment at this moment, and how this episode compares to baseline. Then explain why these findings — the nature and degree taken together — mean that medical treatment in hospital is justified and necessary. Address: what hospital provides that the community cannot; what risks are contained by the inpatient setting; why community treatment is currently insufficient.",

  "q22_risk_if_discharged": "Provide a structured risk assessment with the following headings: RISK TO SELF — include all documented self-harm ideation, threats, and behaviour with causal reasoning about what would happen if discharged; RISK TO OTHERS — document any risk to third parties; RISK OF SELF-NEGLECT AND VULNERABILITY — document any history of poor self-care or functional decline; RISK RELATED TO SUBSTANCE USE — if relevant, describe the relationship between substance use, mental state, and risk; OVERALL RISK SUMMARY — current risk level and key factors that would precipitate deterioration on discharge.",

  "q23_community_risk_management": "Write three paragraphs. COMMUNITY OPTIONS CONSIDERED: describe which community treatment options exist for this patient — CMHT follow-up, crisis team, substance misuse services, psychology, depot medication in the community, or other options. WHY INSUFFICIENT NOW: explain specifically why each option is currently insufficient — address the patient's level of insight, their current engagement with treatment, their history of treatment adherence, whether community measures would be enforceable, and what gap exists between what community services can offer and what this patient currently needs. WHAT NEEDS TO CHANGE: state clearly and specifically what clinical progress is required before community management could be considered safe — this should be a realistic, specific list of what the tribunal would need to see.",

  "q24_recommendations": "This is the section the tribunal will scrutinise most. Write four paragraphs. CURRENT CLINICAL PICTURE: acknowledge any genuine improvement honestly — do not overstate deterioration — but describe the current mental state accurately, including what symptoms or risks remain. State the current trajectory. CRITERIA FOR DETENTION: state explicitly and with reasoning whether the criteria for detention continue to be met. Reference the nature and degree of the disorder, the insight picture, the risk picture, and the treatment requirements. Do not just assert the criteria are met — demonstrate it. CONSEQUENCES OF DISCHARGE NOW: answer the tribunal's core question directly and specifically — if discharged today, what would likely happen? Address each of the following where relevant: likely deterioration in mental state without hospital-based treatment; risk of self-harm or suicidal behaviour; risk of harm to or confrontation with others; risk of disengagement from treatment; resumption of substance use if relevant; vulnerability in the community; likelihood of readmission under emergency powers. Use language such as 'discharge at this stage would, in my clinical judgment, be likely to result in...'. BEFORE DISCHARGE: state clearly what must be achieved before discharge can be considered appropriate — specific, realistic, and clinical.",

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

Return a JSON object with exactly these fields. 

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

  "q8_circumstances_current_admission": "Circumstances leading up to the most recent admission. Include presenting symptoms, behaviour, risk factors, and what led to MHA assessment.",

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


DISCHARGE_SYSTEM = """You are an expert consultant psychiatrist writing a discharge summary.
You write in natural NHS clinical prose — clear, concise, and GP-friendly.
You only use information explicitly present in the notes provided.
You never fabricate clinical details.
You output only valid JSON."""

DISCHARGE_PROMPT = """You are preparing a psychiatric discharge summary from multiple clinical documents.

You have two sources of information. Use BOTH:

STRUCTURED EXTRACTED DATA:
{extracted_data}

FULL CLINICAL NOTES (raw text — use for narrative detail, MSE, risk, medications, and follow-up):
{raw_notes}

WRITING RULES:
- Write in natural NHS consultant prose. No bullet points. No EVIDENCE labels.
- Be concise and GP-friendly — the reader is a GP who needs to understand what happened and what to do next.
- Use temporal language to show how the clinical picture evolved.
- Do not invent missing information. If something is not documented, state this clearly.
- Each section should be one or two focused paragraphs. Do not pad.

Return a JSON object with exactly these fields:

{{
  "patient_name": "First name of patient",

  "reason_for_admission": "Write 2-3 sentences describing why the patient was admitted. Include: the presenting symptoms and behaviour; the precipitating factors (e.g. substance use, medication non-adherence, relapse); and the legal basis for admission if detained. Be specific — name the symptoms, the behaviours, and the circumstances.",

  "clinical_narrative": "Write a clinical narrative of the admission in 3-4 sentences. Cover: what was found on assessment; the diagnosis established or working diagnosis; the key clinical features at presentation; and the initial treatment plan. This is the main explanatory paragraph — it should tell the GP the story of the admission.",

  "progress_on_ward": "Describe how the patient's condition evolved during the admission in 3-4 sentences. Cover: what changed and when; response to treatment; any setbacks or incidents; engagement with the clinical team; and the trajectory towards discharge. Show the journey — not just the endpoint.",

  "mse_on_discharge": "Write a brief mental state examination at the point of discharge in natural prose. Cover: appearance and behaviour; speech; mood (subjective and objective); thoughts (form and content — any residual delusions or ideation); perceptions (hallucinations); cognition if relevant; insight. Write as a clinician would document it — concise and factual.",

  "diagnosis": "State the discharge diagnosis. Use ICD-10 terminology where possible. If there is diagnostic uncertainty, state this. One to two sentences only.",

  "risk_and_crisis_plan": "Summarise the risk picture at discharge and the crisis plan in 3-4 sentences. Cover: current risk to self and others; any specific triggers or warning signs; what the patient should do if they deteriorate; who to contact in a crisis; and any safety netting advice given. Be specific — name the risks and name the plan.",

  "discharge_medications": "List all medications at discharge. For each: name, dose, route, frequency, and any instructions. If there are changes from admission medications, note them. If the patient declined any medication, state this. Write as a medication list in prose: e.g. 'Mirtazapine 30mg oral nocte, Olanzapine 10mg oral nocte...'",

  "follow_up_and_gp_actions": "State the follow-up plan clearly in 3-4 sentences. Cover: who will follow up and when (CMHT, EIT, outpatient clinic); any referrals made; any pending investigations; and specific actions required from the GP — including medication monitoring, blood tests, DVLA if relevant, safeguarding if relevant. Be explicit about what the GP needs to do.",

  "confidence_note": "List any sections where information was limited in the notes and clinician review is especially important."
}}

Critical rules:
- Never leave a field blank — either populate it from the notes or state clearly what is not documented
- Do not invent follow-up plans, referrals, or medications not present in the notes
- The tone should be that of a consultant writing to a GP — professional, clear, and helpful
- Discharge medications must reflect the notes accurately — do not guess doses or routes"""


def generate_discharge(extracted_records, raw_notes, api_key):
    client = OpenAI(api_key=api_key)
    raw_text = "\n\n---\n\n".join(f"NOTE {i+1}:\n{n}" for i,n in enumerate(raw_notes))
    prompt = DISCHARGE_PROMPT.format(
        extracted_data=json.dumps(extracted_records, indent=2, ensure_ascii=False),
        raw_notes=raw_text
    )
    response = client.chat.completions.create(
        model="gpt-4o", temperature=0.0,
        messages=[{"role":"system","content":DISCHARGE_SYSTEM},{"role":"user","content":prompt}]
    )
    output = response.choices[0].message.content.strip()
    if output.startswith("```"):
        output = "\n".join(l for l in output.splitlines() if not l.strip().startswith("```"))
    return json.loads(output)


def render_discharge(ds_data, patient_name, show_debug):
    st.warning("⚠️ AI-assisted draft. Must be reviewed and approved by the responsible clinician before sending.")

    fields = [
        ("Reason for admission",            "reason_for_admission"),
        ("Clinical narrative",              "clinical_narrative"),
        ("Progress on ward",                "progress_on_ward"),
        ("Mental state on discharge",       "mse_on_discharge"),
        ("Diagnosis",                       "diagnosis"),
        ("Risk and crisis plan",            "risk_and_crisis_plan"),
        ("Discharge medications",           "discharge_medications"),
        ("Follow-up and GP actions",        "follow_up_and_gp_actions"),
    ]

    edits = {}
    for label, key in fields:
        val = ds_data.get(key) or "Not documented in available notes — clinician to complete"
        edits[key] = st.text_area(label, value=val, height=120, key=f"ds_{key}")

    if show_debug and ds_data.get("confidence_note"):
        with st.expander("AI confidence note"):
            st.info(ds_data["confidence_note"])

    st.divider()

    plain = f"PSYCHIATRIC DISCHARGE SUMMARY\nPatient: {patient_name}\n\n"
    for label, key in fields:
        plain += f"{label.upper()}\n{edits.get(key, '')}\n\n"
    plain += "---\nAI-ASSISTED DRAFT. Must be reviewed and signed by the responsible clinician before sending.\nGenerated by PsySummarise (research prototype). Not validated for clinical use.\n"

    st.download_button(
        "⬇ Download discharge summary (.txt)",
        data=plain,
        file_name=f"DischargeSummary_{patient_name}.txt",
        mime="text/plain"
    )


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

def generate_s3(extracted_records, raw_notes, risk_assessment, api_key):
    client = OpenAI(api_key=api_key)
    raw_text = "\n\n---\n\n".join(f"NOTE {i+1}:\n{n}" for i,n in enumerate(raw_notes))
    prompt = S3_PROMPT.format(
        extracted_data=json.dumps(extracted_records, indent=2, ensure_ascii=False),
        risk_assessment=json.dumps(risk_assessment, indent=2, ensure_ascii=False),
        raw_notes=raw_text
    )
    response = client.chat.completions.create(
        model="gpt-4o", temperature=0.0,
        messages=[{"role":"system","content":S3_SYSTEM},{"role":"user","content":prompt}]
    )
    output = response.choices[0].message.content.strip()
    if output.startswith("```"):
        output = "\n".join(l for l in output.splitlines() if not l.strip().startswith("```"))
    return json.loads(output)

def generate_tribunal(extracted_records, raw_notes, risk_assessment, tribunal_type, stage, api_key):
    client = OpenAI(api_key=api_key)
    raw_text = "\n\n---\n\n".join(f"NOTE {i+1}:\n{n}" for i,n in enumerate(raw_notes))
    tmpl = TRIBUNAL_INPATIENT_PROMPT if tribunal_type == "Inpatient detention appeal" else TRIBUNAL_CTO_PROMPT
    prompt = tmpl.format(
        extracted_data=json.dumps(extracted_records, indent=2, ensure_ascii=False),
        raw_notes=raw_text,
        risk_assessment=json.dumps(risk_assessment, indent=2, ensure_ascii=False),
        admission_stage=stage
    )
    response = client.chat.completions.create(
        model="gpt-4o", temperature=0.0,
        messages=[{"role":"system","content":TRIBUNAL_SYSTEM},{"role":"user","content":prompt}]
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


import io


ACTION_ICONS = {"start":"\U0001f7e2","continue":"\U0001f535","change":"\U0001f7e1","stop":"\U0001f534"}
ADHERENCE_ICONS = {"good":"\u2705","partial":"\u26a0","poor":"\u274c","unknown":"\u2753"}


def render_risk(risk_data, show_debug, doc_count=None):
    if doc_count:
        st.caption(f"Risk based on {doc_count} document{'s' if doc_count!=1 else ''} analysed")
        if doc_count>=4: st.success("Confidence: High")
        elif doc_count>=2: st.warning("Confidence: Moderate — add more documents for better accuracy")
        else: st.error("Confidence: Low — add more documents for better accuracy")
    st.markdown(f"**Stage:** {risk_data.get('admission_stage','')}")
    if risk_data.get("stage_rationale"): st.caption(risk_data["stage_rationale"])
    domain_labels={"risk_to_self":"Risk to self","risk_to_others":"Risk to others",
        "risk_from_others":"Risk from others / retaliation","risk_of_self_neglect":"Risk of self-neglect",
        "risk_substance_use":"Risk from substance use"}
    for key,label in domain_labels.items():
        d=risk_data.get("domains",{}).get(key,{})
        if not d: continue
        level=d.get("level","?")
        with st.container(border=True):
            c1,c2=st.columns([4,1])
            c1.markdown(f"**{label}**")
            if level=="LOW": c2.success(level)
            elif level=="MODERATE": c2.warning(level)
            else: c2.error(level)
            st.markdown(d.get("tribunal_paragraph",""))
            if show_debug:
                with st.expander("Debug — evidence & scoring"):
                    st.markdown(f"**Past:** {d.get('past_factors','')}")
                    st.markdown(f"**Present:** {d.get('present_factors','')}")
                    st.markdown(f"**Future:** {d.get('future_factors','')}")
                    if d.get("increasing_factors"): st.markdown(f"**Increasing:** {', '.join(d['increasing_factors'])}")
                    if d.get("reducing_factors"): st.markdown(f"**Reducing:** {', '.join(d['reducing_factors'])}")
                    if d.get("evidence"):
                        st.markdown("**Evidence:**")
                        for e in d["evidence"]: st.markdown(f"- {e}")
    st.divider()
    st.markdown("**Overall risk summary**")
    st.markdown(risk_data.get("overall_risk_summary",""))

def render_s3(s3_data, patient_name, show_debug):
    st.warning("\u26a0 AI-assisted draft. Must be reviewed by the responsible clinician before use.")
    st.markdown("*I am approved under section 12 of the Act as having special experience in the diagnosis or treatment of mental disorder.*")
    fields=[("Prior acquaintance with patient","prior_acquaintance"),
            ("Nature of mental disorder","nature_of_disorder"),
            ("Current symptoms","current_symptoms"),
            ("Risk to self","risk_to_self"),("Risk to others","risk_to_others"),
            ("Why informal admission is insufficient","why_informal_insufficient"),
            ("Why community treatment is insufficient","why_community_insufficient"),
            ("Why ongoing inpatient treatment is required","ongoing_treatment_needed"),
            ("Medication history and compliance","medication_history"),
            ("Recommendation","recommendation")]
    edits={}
    for label,key in fields:
        val=s3_data.get(key) or "Not documented — clinician to complete"
        edits[key]=st.text_area(label,value=val,height=100,key=f"s3_{key}")
    if show_debug and s3_data.get("confidence_note"):
        with st.expander("AI confidence note"): st.info(s3_data["confidence_note"])
    st.divider()
    plain=f"MEDICAL RECOMMENDATION FOR ADMISSION FOR TREATMENT (SECTION 3 MHA 1983 - FORM A8)\nPatient: {patient_name}\n\n"
    plain+="I am approved under section 12 of the Act as having special experience in the diagnosis or treatment of mental disorder.\n\n"
    for label,key in fields: plain+=f"{label.upper()}\n{edits.get(key,'')}\n\n"
    plain+="---\nAI-ASSISTED DRAFT. Must be reviewed and signed by an approved clinician.\nGenerated by PsySummarise (research prototype). Not validated for clinical use.\n"
    st.download_button("\u2b07 Download Section 3 draft (.txt)",data=plain,
                       file_name=f"Section3_{patient_name}.txt",mime="text/plain")

def render_tribunal(tr_data, patient_name, tribunal_type_key, show_debug):
    st.warning("\u26a0 AI-assisted draft. Must be reviewed by the Responsible Clinician before tribunal submission.")
    s_inp=[("3. Factors affecting hearing","q3_factors_affecting_hearing"),
           ("4. Adjustments","q4_adjustments"),("5. Forensic history","q5_forensic_history"),
           ("6. Previous MH involvement","q6_previous_mh_involvement"),
           ("7. Reasons for previous admissions","q7_reasons_previous_admissions"),
           ("8. Circumstances of current admission","q8_circumstances_current_admission"),
           ("9. Mental disorder present?","q9_mental_disorder_present"),
           ("10. Diagnosis","q10_diagnosis"),("11. Learning disability?","q11_learning_disability"),
           ("12. Mental disorder requiring detention?","q12_detention_required"),
           ("13. Treatment available","q13_treatment"),
           ("14. Strengths and positive factors","q14_strengths"),
           ("15. Current progress, behaviour, capacity and insight","q15_current_progress"),
           ("16. Medication compliance and future willingness","q16_medication_compliance"),
           ("17. MCA consideration","q17_mca_consideration"),
           ("18. Incidents of harm or threats","q18_incidents_self_harm_others"),
           ("19. Property damage","q19_property_damage"),
           ("20. Is detention justified — health, safety or protection of others?","q20_section2_detention_justified"),
           ("21. Nature, degree and necessity of inpatient treatment","q21_treatment_in_hospital_justified"),
           ("22. Risk of dangerous behaviour if discharged","q22_risk_if_discharged"),
           ("23. Community risk management","q23_community_risk_management"),
           ("24. Recommendations","q24_recommendations")]
    s_cto=[("2. Capacity for hearing","q2_capacity_hearing"),
           ("3. Factors affecting hearing","q3_factors_affecting_hearing"),
           ("4. Adjustments","q4_adjustments"),("5. Forensic history","q5_forensic_history"),
           ("6. Previous MH involvement","q6_previous_mh_involvement"),
           ("7. Reasons for previous admissions","q7_reasons_previous_admissions"),
           ("8. Circumstances of current admission","q8_circumstances_current_admission"),
           ("9. Mental disorder and diagnosis","q9_mental_disorder"),
           ("10. Learning disability?","q10_learning_disability"),
           ("11. CTO treatment appropriate?","q11_treatment_appropriate"),
           ("12. Treatment details","q12_treatment_details"),
           ("13. Strengths and positive factors","q13_strengths"),
           ("14. Current progress in community","q14_current_progress"),
           ("15. Medication compliance","q15_medication_compliance"),
           ("16. MCA consideration","q16_mca_consideration"),
           ("17. Incidents of harm or threats","q17_incidents"),
           ("18. Treatment necessary under CTO?","q18_treatment_necessary"),
           ("19. Risk if CTO rescinded","q19_risk_if_cto_rescinded"),
           ("20. Community risk management","q20_community_risk_management"),
           ("21. Recommendations","q21_recommendations")]
    sections=s_inp if tribunal_type_key=="Inpatient detention appeal" else s_cto
    edits={}
    for label,key in sections:
        val=tr_data.get(key) or "Not documented — clinician to complete"
        edits[key]=st.text_area(label,value=val,height=120,key=f"tr_{key}")
    if show_debug and tr_data.get("confidence_note"):
        with st.expander("AI confidence note"): st.info(tr_data["confidence_note"])
    st.divider()
    plain=f"RESPONSIBLE CLINICIAN'S REPORT FOR MENTAL HEALTH TRIBUNAL\nTribunal type: {tr_data.get('tribunal_type','')}\nPatient: {patient_name}\nRC: {tr_data.get('rc_name','')}\n\n"
    for label,key in sections: plain+=f"{label.upper()}\n{edits.get(key,'')}\n\n"
    plain+="---\nAI-ASSISTED DRAFT. Must be reviewed and signed by the Responsible Clinician.\nGenerated by PsySummarise (research prototype). Not validated for clinical use.\n"
    st.download_button("\u2b07 Download tribunal report (.txt)",data=plain,
                       file_name=f"TribunalReport_{patient_name}.txt",mime="text/plain")

def add_notes_widget(key_prefix, notes_key, extracted_key, api_key):
    input_mode=st.radio("Input method",["Paste text","Upload files (PDF / Word)"],
                        horizontal=True,key=f"{key_prefix}_mode")
    if input_mode=="Paste text":
        note_text=st.text_area("Paste note here",height=160,
            placeholder="Paste a ward round note, admission summary, MDT review, or discharge letter...",
            key=f"{key_prefix}_paste")
        c1,c2=st.columns([3,1])
        with c1: add=st.button("Add note \u2192",key=f"{key_prefix}_add")
        with c2:
            if st.button("Clear all",key=f"{key_prefix}_clear"):
                st.session_state[notes_key]=[]; st.session_state[extracted_key]=[]; st.rerun()
        if add:
            if not api_key: st.error("Please enter your OpenAI API key in the sidebar.")
            elif not note_text or not note_text.strip(): st.warning("Please paste a note above.")
            else:
                with st.spinner(f"Extracting note {len(st.session_state[notes_key])+1}..."):
                    try:
                        extracted=extract_note(note_text,api_key)
                        st.session_state[notes_key].append(note_text)
                        st.session_state[extracted_key].append(extracted)
                        st.success(f"Note {len(st.session_state[notes_key])} added — {extracted.get('patient_id','?')}/{extracted.get('doc_id','?')}")
                    except Exception as e: st.error(f"Extraction failed: {e}")
    else:
        st.info("\u2139 PDFs must contain selectable text. Scanned/image-only PDFs cannot be read.")
        uploaded=st.file_uploader("Upload one or more files",type=["pdf","docx"],
                                  accept_multiple_files=True,key=f"{key_prefix}_multi")
        c1,c2=st.columns([3,1])
        with c1:
            lbl=f"Extract {len(uploaded)} file(s) \u2192" if uploaded else "Extract files \u2192"
            add_files=st.button(lbl,key=f"{key_prefix}_addfiles",disabled=not uploaded)
        with c2:
            if st.button("Clear all",key=f"{key_prefix}_clear2"):
                st.session_state[notes_key]=[]; st.session_state[extracted_key]=[]; st.rerun()
        if add_files and uploaded:
            if not api_key: st.error("Please enter your OpenAI API key in the sidebar.")
            else:
                prog=st.progress(0)
                for i,f in enumerate(uploaded):
                    text,err=extract_text_from_file(f)
                    if err=="pdf_empty": st.error(f"{f.name}: No text — may be scanned."); continue
                    elif err: st.error(f"{f.name}: {err}"); continue
                    with st.spinner(f"Extracting {f.name}..."):
                        try:
                            extracted=extract_note(text,api_key)
                            st.session_state[notes_key].append(text)
                            st.session_state[extracted_key].append(extracted)
                            st.success(f"{f.name} \u2192 {extracted.get('patient_id','?')}/{extracted.get('doc_id','?')}")
                        except Exception as e: st.error(f"{f.name}: {e}")
                    prog.progress((i+1)/len(uploaded))
                prog.empty()
    if st.session_state[extracted_key]:
        st.markdown(f"**{len(st.session_state[extracted_key])} note(s) loaded:**")
        for i,r in enumerate(st.session_state[extracted_key]):
            st.caption(f"{i+1}. {r.get('patient_id','?')}/{r.get('doc_id','?')} — "
                       f"{len(r.get('medications',[]))} med(s), {len(r.get('diagnoses',[]))} dx, "
                       f"{len(r.get('admissions',[]))} admission event(s)")

def stage_selector(key):
    options=["Acute admission (first days)",
             "Mid-admission (settled but ongoing symptoms)",
             "Discharge-ready (nearing discharge)"]
    stage=st.selectbox("Admission stage",options,index=1,key=f"{key}_stage")
    override=st.text_input("Override / add detail (optional)",
        placeholder="e.g. Day 12, refusing antipsychotics, partial insight",key=f"{key}_override")
    return override.strip() if override.strip() else stage

# ── App layout ────────────────────────────────────────────────────────────────
st.title("\U0001f9e0 PsySummarise")
st.caption("Structured extraction from psychiatric documentation \u00b7 Research prototype \u00b7 Not validated for clinical use")
st.divider()

for key in ["s3_notes","s3_extracted","tr_notes","tr_extracted","ds_notes","ds_extracted"]:
    if key not in st.session_state: st.session_state[key]=[]

with st.sidebar:
    st.markdown("### \u2699 Settings")
    api_key=st.text_input("OpenAI API Key",type="password",placeholder="sk-...")
    st.divider()
    show_debug=st.toggle("Debug mode",value=False,
        help="Shows evidence citations, risk scoring, and AI confidence notes")
    st.divider()
    st.caption("PsySummarise uses GPT-4o. Research prototype. Not validated for clinical use.")

tab1,tab2,tab3,tab4=st.tabs(["\U0001f4c4 Single note extraction",
                              "\U0001f4cb Section 3 recommendation",
                              "\u2696 Tribunal report",
                              "\U0001f4cb Discharge summary"])

with tab1:
    note_text=note_input_widget("t1") if "note_input_widget" in dir() else st.text_area("Paste note",height=180,key="t1_paste")
    if st.button("Extract structured data \u2192",key="extract_single"):
        if not api_key: st.error("Please enter your OpenAI API key in the sidebar.")
        elif not note_text or not note_text.strip(): st.warning("Please provide a note above.")
        else:
            with st.spinner("Extracting..."):
                try:
                    result=extract_note(note_text,api_key)
                    st.success("Extraction complete.")
                    st.divider()
                    s_tab,j_tab=st.tabs(["Clinical summary","Raw JSON"])
                    with s_tab: render_clinical_summary(result)
                    with j_tab:
                        st.json(result)
                        st.download_button("\u2b07 Download JSON",
                            data=json.dumps(result,indent=2,ensure_ascii=False),
                            file_name=f"psysummarise_{result.get('patient_id','output')}.json",
                            mime="application/json")
                except Exception as e: st.error(f"Error: {e}")

with tab2:
    st.markdown("### Build a Section 3 recommendation from multiple notes")
    st.caption("Add each clinical document one at a time. Select admission stage before generating.")
    add_notes_widget("s3","s3_notes","s3_extracted",api_key)
    if st.session_state.s3_extracted:
        st.divider()
        stage=stage_selector("s3")
        if st.button("Generate Section 3 recommendation \u2192",key="gen_s3"):
            if not api_key: st.error("Please enter your OpenAI API key in the sidebar.")
            else:
                risk_result={}
                with st.spinner("Computing risk assessment..."):
                    try: risk_result=compute_risk(st.session_state.s3_extracted,st.session_state.s3_notes,stage,api_key)
                    except Exception as e: st.warning(f"Risk assessment failed: {e}")
                with st.spinner("Generating Section 3 recommendation..."):
                    try:
                        s3_result=generate_s3(st.session_state.s3_extracted,st.session_state.s3_notes,risk_result,api_key)
                        st.success("Draft generated. Review all fields carefully.")
                        st.divider()
                        patient_name=s3_result.get("patient_name") or st.session_state.s3_extracted[0].get("patient_id","Patient")
                        r_tab,d_tab,src_tab=st.tabs(["Risk assessment","Draft recommendation","Source data"])
                        with r_tab: render_risk(risk_result,show_debug,len(st.session_state.s3_extracted))
                        with d_tab: render_s3(s3_result,patient_name,show_debug)
                        with src_tab: st.json(st.session_state.s3_extracted)
                    except Exception as e: st.error(f"Error: {e}")
    else: st.info("Add at least one note above to begin.")

with tab3:
    st.markdown("### Generate a tribunal report from multiple notes")
    st.caption("Add each clinical document one at a time. Select tribunal type and admission stage before generating.")
    tribunal_type=st.radio("Tribunal type",
        ["Inpatient detention appeal (Section 2 / Section 3)","CTO appeal"],horizontal=True)
    tribunal_type_key="Inpatient detention appeal" if "Inpatient" in tribunal_type else "CTO appeal"
    add_notes_widget("tr","tr_notes","tr_extracted",api_key)
    if st.session_state.tr_extracted:
        st.divider()
        stage=stage_selector("tr")
        if st.button("Generate tribunal report \u2192",key="gen_tribunal"):
            if not api_key: st.error("Please enter your OpenAI API key in the sidebar.")
            else:
                tr_risk={}
                with st.spinner("Computing risk assessment..."):
                    try: tr_risk=compute_risk(st.session_state.tr_extracted,st.session_state.tr_notes,stage,api_key)
                    except Exception as e: st.warning(f"Risk assessment failed: {e}")
                with st.spinner("Generating tribunal report — this may take 30-60 seconds..."):
                    try:
                        tr_result=generate_tribunal(st.session_state.tr_extracted,st.session_state.tr_notes,
                            tr_risk,tribunal_type_key,stage,api_key)
                        st.success("Draft generated. Review all sections carefully.")
                        st.divider()
                        patient_name=tr_result.get("patient_name") or st.session_state.tr_extracted[0].get("patient_id","Patient")
                        r_tab,d_tab,src_tab=st.tabs(["Risk assessment","Draft report","Source data"])
                        with r_tab: render_risk(tr_risk,show_debug,len(st.session_state.tr_extracted))
                        with d_tab: render_tribunal(tr_result,patient_name,tribunal_type_key,show_debug)
                        with src_tab: st.json(st.session_state.tr_extracted)
                    except Exception as e: st.error(f"Error: {e}")
    else: st.info("Add at least one note above to begin.")

with tab4:
    st.markdown("### Generate a discharge summary from multiple notes")
    st.caption("Add each clinical document — ward rounds, MDT notes, admission summary, discharge letter. The more notes, the better the output.")
    add_notes_widget("ds","ds_notes","ds_extracted",api_key)
    if st.session_state.ds_extracted:
        st.divider()
        if st.button("Generate discharge summary \u2192",key="gen_ds"):
            if not api_key: st.error("Please enter your OpenAI API key in the sidebar.")
            else:
                with st.spinner("Generating discharge summary — this may take 20-40 seconds..."):
                    try:
                        ds_result=generate_discharge(st.session_state.ds_extracted,st.session_state.ds_notes,api_key)
                        st.success("Draft generated. Review all sections carefully.")
                        st.divider()
                        patient_name=ds_result.get("patient_name") or st.session_state.ds_extracted[0].get("patient_id","Patient")
                        d_tab,src_tab=st.tabs(["Draft summary","Source data"])
                        with d_tab: render_discharge(ds_result,patient_name,show_debug)
                        with src_tab: st.json(st.session_state.ds_extracted)
                    except Exception as e: st.error(f"Error: {e}")
    else: st.info("Add at least one note above to begin.")
