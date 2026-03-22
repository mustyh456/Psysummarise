import streamlit as st
import json
import os
import io
from openai import OpenAI

st.set_page_config(page_title="PsySummarise", page_icon="🧠", layout="wide")

# ── File parsing ──────────────────────────────────────────────────────────────
def extract_text_from_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            if not text.strip():
                return None, "pdf_empty"
            return text, None
        except Exception as e:
            return None, str(e)
    elif name.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(uploaded_file.read()))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return text, None
        except Exception as e:
            return None, str(e)
    return None, "unsupported"

# ── Prompts ───────────────────────────────────────────────────────────────────
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
  * clear = explicitly stated in the note
  * unclear = mentioned but ambiguous
  * not_stated = not mentioned at all in the note
- mdt_agreement must be one of: agree, disagree, unclear, not_stated
- For medication route and schedule: infer from drug name if not explicitly stated
  * Mirtazapine → route: oral, schedule: nocte
  * Olanzapine oral → route: oral
  * Lorazepam PRN → route: oral, schedule: PRN
  * Zopiclone → route: oral, schedule: nocte PRN
  * Depot injections → route: IM depot
  * If genuinely unclear, use null
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

Analyse the following clinical data and produce a structured risk assessment across all five domains.

EXTRACTED CLINICAL DATA:
{extracted_data}

RAW CLINICAL NOTES:
{raw_notes}

ADMISSION STAGE: {admission_stage}

Use these exact risk level definitions:
- LOW: no recent incidents, no active symptoms driving risk, adequate insight and engagement
- MODERATE: recent risk history OR ongoing symptoms OR partial insight, currently mitigated by structure/support
- HIGH: recent serious incident PLUS active symptoms AND poor insight, impulsivity, or inability to engage safely

Risk must be anchored across time — always address past, present, and future.
Current inpatient risk should generally be lower than likely community risk.
For mid-admission patients who are settled: do not rate risk as HIGH unless there is a very recent serious incident — use MODERATE with appropriate modifiers.
For discharge-ready patients: risk levels should reflect likely community risk, not inpatient risk.

Return a JSON object with exactly this structure:
{{
  "patient_name": "First name",
  "admission_stage": "{admission_stage}",
  "domains": {{
    "risk_to_self": {{
      "level": "LOW/MODERATE/HIGH",
      "past_factors": "historical incidents and risk events",
      "present_factors": "current risk in inpatient setting",
      "future_factors": "likely risk if discharged prematurely",
      "increasing_factors": ["list of factors increasing risk"],
      "reducing_factors": ["list of factors reducing risk"],
      "evidence": ["specific clinical events from notes"],
      "tribunal_paragraph": "Clean tribunal-ready paragraph. No NOTE citations. Embed evidence naturally. Calibrate to admission stage. Distinguish current inpatient risk from likely community risk."
    }},
    "risk_to_others": {{
      "level": "LOW/MODERATE/HIGH",
      "past_factors": "historical risk to others",
      "present_factors": "current risk in inpatient setting",
      "future_factors": "likely risk if discharged",
      "increasing_factors": [],
      "reducing_factors": [],
      "evidence": [],
      "tribunal_paragraph": "Clean tribunal paragraph. Calibrated to stage."
    }},
    "risk_from_others": {{
      "level": "LOW/MODERATE/HIGH",
      "past_factors": "any history of being targeted, vulnerability to exploitation, or retaliation risk",
      "present_factors": "current vulnerability or retaliation risk",
      "future_factors": "risk of retaliation or exploitation if discharged",
      "increasing_factors": [],
      "reducing_factors": [],
      "evidence": [],
      "tribunal_paragraph": "Address whether the patient's behaviour or beliefs could provoke retaliation from others, or whether they are vulnerable to exploitation. Note if confrontational beliefs increase risk of others retaliating against them."
    }},
    "risk_of_self_neglect": {{
      "level": "LOW/MODERATE/HIGH",
      "past_factors": "history of self-neglect, poor self-care, social decline",
      "present_factors": "current self-care in inpatient setting",
      "future_factors": "risk of neglect if discharged without support",
      "increasing_factors": [],
      "reducing_factors": [],
      "evidence": [],
      "tribunal_paragraph": "Clean tribunal paragraph."
    }},
    "risk_substance_use": {{
      "level": "LOW/MODERATE/HIGH",
      "past_factors": "substance use history and its clinical impact",
      "present_factors": "current substance use or abstinence in hospital",
      "future_factors": "risk of resumption and impact on mental state if discharged",
      "increasing_factors": [],
      "reducing_factors": [],
      "evidence": [],
      "tribunal_paragraph": "Clean tribunal paragraph."
    }}
  }},
  "overall_risk_summary": "One paragraph overall risk summary calibrated to admission stage. State current inpatient risk level and likely community risk level separately. Use language such as: 'Risk is currently reduced within the structured inpatient setting; however, there remains a [level] risk of deterioration if discharged prematurely, given [key factors].'",
  "stage_rationale": "Brief explanation of why risk levels are calibrated to this admission stage"
}}"""

S3_SYSTEM = """You are an expert psychiatrist drafting a Section 3 MHA recommendation.
You write clearly and clinically. You only use information present in the notes.
You output only valid JSON."""

S3_PROMPT = """Based on the following clinical data and risk assessment, draft a Section 3 MHA recommendation (Form A8).

Extracted clinical data:
{extracted_data}

Pre-computed risk assessment:
{risk_assessment}

Raw notes:
{raw_notes}

Return a JSON object with exactly these fields.
Write clean clinical prose throughout. Do NOT include NOTE citations or EVIDENCE labels.
Embed evidence naturally as a clinician would write. Do not use "EVIDENCE:" labels.

{{
  "patient_name": "First name",
  "prior_acquaintance": "Not documented — clinician to complete",
  "nature_of_disorder": "Use statutory phrase 'mental disorder of a nature and degree which makes it appropriate for the patient to receive medical treatment in a hospital'. Follow with diagnosis and clinical picture.",
  "current_symptoms": "Current symptoms and MSE findings in clinical prose.",
  "risk_to_self": "Draw from pre-computed risk assessment. Clean prose, no citations.",
  "risk_to_others": "Draw from pre-computed risk assessment. Clean prose.",
  "why_informal_insufficient": "Insight, capacity, likely compliance if not detained.",
  "why_community_insufficient": "Why community treatment cannot safely manage this presentation.",
  "ongoing_treatment_needed": "Why ongoing inpatient treatment is required — mental health optimisation, MDT input, medication monitoring, developing insight, safe discharge planning.",
  "medication_history": "All medications with doses and routes. Compliance with each. Explicitly state whether IM medication has been required. Address any medications refused and reasons.",
  "recommendation": "Recommend Section 3 detention framed around: further inpatient treatment required for optimisation of mental health, appropriate clinical management, and safe planned discharge when clinically indicated.",
  "confidence_note": "Sections needing most clinician attention."
}}"""

TRIBUNAL_SYSTEM = """You are an expert consultant psychiatrist preparing a statutory Mental Health Tribunal report.
You write clearly and with appropriate legal precision.
You only use information present in the notes.
You output only valid JSON."""

TRIBUNAL_INPATIENT_PROMPT = """Prepare a Responsible Clinician's report for an inpatient Mental Health Tribunal.
The patient is detained and appealing against detention under Section 2 or Section 3.

EXTRACTED DATA: {extracted_data}
RAW NOTES: {raw_notes}
PRE-COMPUTED RISK ASSESSMENT: {risk_assessment}
ADMISSION STAGE: {admission_stage}

CRITICAL FORMATTING RULES:
- Write clean clinical prose throughout — NO "EVIDENCE: NOTE X" citations anywhere
- Embed evidence naturally: write "He threatened to cut his throat" not "EVIDENCE: NOTE 1"
- Each section should read as a consultant clinician wrote it
- Calibrate risk to the admission stage provided
- Sections 3 and 4 are about physical/cognitive accessibility ONLY — not mental state symptoms

Return JSON with exactly these fields:

{{
  "patient_name": "First name",
  "rc_name": "Not documented — clinician to complete",
  "tribunal_type": "Inpatient detention appeal",

  "q3_factors_affecting_hearing": "Intellectual disabilities, physical disabilities, sensory impairments, or communication difficulties affecting tribunal participation only. If none: 'There are no known factors that would affect the patient's ability to participate in a tribunal hearing.'",

  "q4_adjustments": "Physical or communication adjustments only. If none: 'No specific adjustments are required at this time.'",

  "q5_forensic_history": "Formal forensic history. If none, state clearly. Add brief contextual note about risk-relevant behaviours documented in the notes even if not resulting in formal proceedings.",

  "q6_previous_mh_involvement": "All previous MH involvement with dates. Draw from raw notes — include primary care history. If first admission state clearly.",

  "q7_reasons_previous_admissions": "Reasons for previous admissions. If first: 'This is the patient's first psychiatric admission.'",

  "q8_circumstances_current_admission": "Detailed multi-paragraph narrative: background; prodromal period; specific symptoms and behaviours; full risk picture including ALL documented risk events such as suicidal ideation, threats, weapons, confrontational behaviour; what led to MHA assessment; legal basis. Do not omit risk events.",

  "q9_mental_disorder_present": "Answer 'Yes.' or 'No.' only. Do not include diagnosis or symptoms here — those belong in section 10.",

  "q10_diagnosis": "State diagnosis. One sentence clinical basis. In substance-related cases address substance-induced versus primary disorder.",

  "q11_learning_disability": "Yes or no with brief explanation.",

  "q12_detention_required": "Detailed reasoning: nature of disorder; insight and attitude to treatment; specific risk behaviours making community management unsafe; why hospital is necessary. Explicit causal chain: mental disorder leading to impaired insight leading to risk behaviour leading to need for detention.",

  "q13_treatment": "PHARMACOLOGICAL: all medications with doses and routes, compliance with each, any medications offered but declined with reasons, whether IM medication has been required during admission. NON-PHARMACOLOGICAL: psychology, OT, substance misuse referrals — for each state whether offered, engaged with, or declined. COMMUNITY TEAM INVOLVEMENT: only include this if a referral to EIT, CMHT, or other community team is documented in the notes — if so, state the outcome including if declined or not taken on; do not mention community referrals if not documented. ENGAGEMENT: overall engagement and attitude to treatment. PLANNED: planned treatment going forward.",

  "q14_strengths": "All strengths — engagement with staff, compliance, calm and cooperative, no behavioural disturbance, emerging insight, social support, protective factors.",

  "q15_current_progress": "PRESENTATION AT ADMISSION: mental state on day of admission. PROGRESS DURING ADMISSION: how mental state evolved. CURRENT NURSING OBSERVATIONS: recent behaviour, engagement, sleep, appetite. CAPACITY: assessment findings with date. INSIGHT: current level. OVERALL TRAJECTORY: improving, stable, or deteriorating.",

  "q16_medication_compliance": "List every medication with name, dose, route, and schedule — infer route and schedule from drug name if not explicitly stated (e.g. Mirtazapine is oral nocte, Lorazepam PRN is oral). State compliance with each medication. Explicitly state whether IM medication has been required — if not: 'No IM medication has been required during this admission.' Address any medications offered but declined and the patient's reasons. Comment on likely future willingness to accept treatment. Note if any refusal appears capacitous but may be influenced by ongoing symptoms.",

  "q17_mca_consideration": "State capacity assessment finding with date. If patient has capacity: 'The Mental Health Act remains the appropriate legal framework as the patient meets the criteria for detention regardless of their capacity to consent to treatment.' Keep this brief.",

  "q18_incidents_self_harm_others": "ALL incidents from BOTH during admission and prior to admission. Include suicidal ideation, threats of self-harm, actual self-harm, weapons, confrontational behaviour, threatening behaviour. This section must be complete.",

  "q19_property_damage": "Any property damage or threats. If none state clearly. Note escalating behaviours falling short of damage such as shouting or banging on walls.",

  "q20_section2_detention_justified": "HEALTH: why detention necessary for health. SAFETY: why necessary for safety with reference to specific risk events. PROTECTION OF OTHERS: third-party risk including retaliation risk. WHY COMMUNITY IS INSUFFICIENT: specific reasoning.",

  "q21_treatment_in_hospital_justified": "Why inpatient treatment is justified. Cover: monitoring needs; risks contained by inpatient setting including impulsive behaviour and risk of retaliation from others if patient acts on delusional beliefs in the community; why community treatment is currently insufficient.",

  "q22_risk_if_discharged": "Draw from the pre-computed risk assessment. Structure as: RISK TO SELF; RISK TO OTHERS; RISK FROM OTHERS AND RETALIATION; RISK OF SELF-NEGLECT; RISK FROM SUBSTANCE USE; OVERALL SUMMARY. Use calibrated tribunal phrasing throughout — for example 'moderate risk of deterioration and recurrence of harmful behaviours' rather than simply 'moderate risk'. For mid-admission patients who are settled, state that risk is currently reduced in the structured inpatient setting but there remains a moderate risk of deterioration and recurrence if discharged prematurely.",

  "q23_community_risk_management": "ALTERNATIVES CONSIDERED: informal admission and why not appropriate, CRHT involvement, CMHT follow-up, CTO and whether applicable at this stage. WHY CURRENTLY INSUFFICIENT: enforceability of treatment, engagement reliability, insight level. WHAT WOULD NEED TO CHANGE for community management to become safe.",

  "q24_recommendations": "Full recommendation covering: current symptoms and mental state; risk picture; insight and treatment compliance; trajectory of improvement. Explicitly state whether the criteria for detention continue to be met and why. State what further progress is required before discharge would be clinically appropriate. Acknowledge improvement where present but balance this against ongoing clinical need.",

  "confidence_note": "Sections needing most clinician attention."
}}"""

TRIBUNAL_CTO_PROMPT = """Prepare a Responsible Clinician's report for a CTO Mental Health Tribunal.
The patient is in the community under a CTO and is appealing against it.

EXTRACTED DATA: {extracted_data}
RAW NOTES: {raw_notes}
PRE-COMPUTED RISK ASSESSMENT: {risk_assessment}
ADMISSION STAGE: {admission_stage}

CRITICAL: Write clean clinical prose — NO "EVIDENCE: NOTE X" citations. Embed evidence naturally.
Sections 3 and 4 are about physical/cognitive accessibility ONLY.

Return JSON with exactly these fields:

{{
  "patient_name": "First name",
  "rc_name": "Not documented — clinician to complete",
  "tribunal_type": "CTO appeal",

  "q2_capacity_hearing": "Capacity to attend tribunal and be represented. State finding with date.",
  "q3_factors_affecting_hearing": "Physical/cognitive accessibility only. If none: 'There are no known factors that would affect the patient's ability to participate.'",
  "q4_adjustments": "Physical/communication adjustments. If none: 'No specific adjustments required.'",
  "q5_forensic_history": "Formal forensic history. Contextual note on risk-relevant behaviours.",
  "q6_previous_mh_involvement": "All previous MH involvement with dates. Draw from raw notes.",
  "q7_reasons_previous_admissions": "Reasons for previous admissions.",
  "q8_circumstances_current_admission": "Detailed narrative of circumstances leading to admission and subsequent CTO. Include full risk picture.",
  "q9_mental_disorder": "'Yes.' Diagnosis and key symptoms in one sentence. Whether nature and degree warrant ongoing CTO.",
  "q10_learning_disability": "Yes or no.",
  "q11_treatment_appropriate": "Why ongoing CTO treatment remains necessary and appropriate.",
  "q12_treatment_details": "PHARMACOLOGICAL: medications with doses, compliance, depot schedule. NON-PHARMACOLOGICAL: community input, psychology, social work. ENGAGEMENT: with CTO conditions. PLANNED: future treatment.",
  "q13_strengths": "All positive factors and protective factors.",
  "q14_current_progress": "Current progress in community — mental state, compliance, insight, trajectory.",
  "q15_medication_compliance": "Compliance with depot or oral medication under CTO. Likely compliance if CTO rescinded.",
  "q16_mca_consideration": "Capacity finding with date. Brief explanation of MHA vs MCA.",
  "q17_incidents": "ALL incidents from admission and community. Must be complete.",
  "q18_treatment_necessary": "Why CTO treatment necessary for health, safety, protection of others.",
  "q19_risk_if_cto_rescinded": "Draw from risk assessment. RISK TO SELF; RISK TO OTHERS; RISK FROM OTHERS; RISK OF SELF-NEGLECT; RISK FROM NON-ADHERENCE TO MEDICATION; OVERALL SUMMARY.",
  "q20_community_risk_management": "How risks are managed under CTO. What would happen without compulsory powers — address enforceability and medication adherence.",
  "q21_recommendations": "Full recommendation — diagnosis, mental state, risk, insight, compliance, why CTO criteria remain met, consequences of rescission.",
  "confidence_note": "Sections needing most clinician attention."
}}"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_schema():
    if os.path.exists("extraction_schema.json"):
        with open("extraction_schema.json") as f:
            return json.load(f)
    return {}


def call_api(prompt, system, api_key):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o", temperature=0.0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    )
    output = response.choices[0].message.content.strip()
    if output.startswith("```"):
        output = "\n".join(l for l in output.splitlines() if not l.strip().startswith("```"))
    return json.loads(output)


def extract_note(note_text, api_key):
    schema = load_schema()
    prompt = EXTRACTION_PROMPT.format(schema=json.dumps(schema, indent=2), note=note_text)
    return call_api(prompt, SYSTEM_PROMPT, api_key)


def raw_notes_text(notes):
    return "\n\n---\n\n".join(f"NOTE {i+1}:\n{n}" for i, n in enumerate(notes))


def compute_risk(extracted, notes, stage, api_key):
    prompt = RISK_ENGINE_PROMPT.format(
        extracted_data=json.dumps(extracted, indent=2, ensure_ascii=False),
        raw_notes=raw_notes_text(notes),
        admission_stage=stage
    )
    return call_api(prompt, SYSTEM_PROMPT, api_key)


def generate_s3(extracted, notes, risk, api_key):
    prompt = S3_PROMPT.format(
        extracted_data=json.dumps(extracted, indent=2, ensure_ascii=False),
        risk_assessment=json.dumps(risk, indent=2, ensure_ascii=False),
        raw_notes=raw_notes_text(notes)
    )
    return call_api(prompt, S3_SYSTEM, api_key)


def generate_tribunal(extracted, notes, risk, tribunal_type, stage, api_key):
    tmpl = TRIBUNAL_INPATIENT_PROMPT if tribunal_type == "Inpatient detention appeal" else TRIBUNAL_CTO_PROMPT
    prompt = tmpl.format(
        extracted_data=json.dumps(extracted, indent=2, ensure_ascii=False),
        raw_notes=raw_notes_text(notes),
        risk_assessment=json.dumps(risk, indent=2, ensure_ascii=False),
        admission_stage=stage
    )
    return call_api(prompt, TRIBUNAL_SYSTEM, api_key)


ACTION_ICONS = {"start": "🟢", "continue": "🔵", "change": "🟡", "stop": "🔴"}
ADHERENCE_ICONS = {"good": "✅", "partial": "⚠️", "poor": "❌", "unknown": "❓"}


def render_clinical_summary(data):
    st.markdown(f"## {data.get('patient_id') or 'Unknown'}")
    st.caption(f"Document: {data.get('doc_id') or 'Unknown'}")
    for adm in data.get("admissions", []):
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Legal status", adm.get("legal_status") or "?")
            c2.metric("Event type", adm.get("event_type") or "?")
            c3.metric("Date", adm.get("event_date") or "unknown")
            if adm.get("reason"): st.markdown(f"**Reason:** {adm['reason']}")
            if adm.get("evidence_quote"): st.caption(f'📎 *"{adm["evidence_quote"]}"*')
    for dx in data.get("diagnoses", []):
        with st.container(border=True):
            st.markdown(f"**{dx.get('dx_label') or 'Unknown'}**")
            c = st.columns(4)
            if dx.get("icd10"): c[0].metric("ICD-10", dx["icd10"])
            if dx.get("status"): c[1].metric("Status", dx["status"])
            if dx.get("mdt_agreement"): c[2].metric("MDT", dx["mdt_agreement"])
            if dx.get("certainty"): c[3].metric("Certainty", dx["certainty"])
            if dx.get("evidence_quote"): st.caption(f'📎 *"{dx["evidence_quote"]}"*')
    for med in data.get("medications", []):
        action = med.get("action") or "?"
        adh = med.get("adherence") or "unknown"
        with st.container(border=True):
            st.markdown(f"{ACTION_ICONS.get(action,'⚪')} **{action.upper()}** — {med.get('med_name') or '?'} {med.get('dose_text') or ''}")
            c = st.columns(4)
            if med.get("route"): c[0].metric("Route", med["route"])
            if med.get("schedule"): c[1].metric("Schedule", med["schedule"])
            c[2].metric("Adherence", f"{ADHERENCE_ICONS.get(adh,'❓')} {adh}")
            c[3].metric("Certainty", med.get("certainty") or "?")
            if med.get("response"): st.markdown(f"**Response:** {med['response']}")
            if med.get("evidence_quote"): st.caption(f'📎 *"{med["evidence_quote"]}"*')


def render_risk(risk_data, show_debug, doc_count=None):
    stage = risk_data.get("admission_stage","")
    rationale = risk_data.get("stage_rationale","")
    if doc_count:
        st.caption(f"Risk based on {doc_count} document{'s' if doc_count != 1 else ''} analysed")
    # Confidence indicator based on doc count
    if doc_count:
        if doc_count >= 4:
            conf_label, conf_color = "High confidence", "success"
        elif doc_count >= 2:
            conf_label, conf_color = "Moderate confidence", "warning"
        else:
            conf_label, conf_color = "Low confidence — add more documents for better accuracy", "error"
        if conf_color == "success": st.success(f"Confidence: {conf_label}")
        elif conf_color == "warning": st.warning(f"Confidence: {conf_label}")
        else: st.error(f"Confidence: {conf_label}")
    st.markdown(f"**Stage:** {stage}  \n*{rationale}*")
    domain_labels = {
        "risk_to_self": "Risk to self",
        "risk_to_others": "Risk to others",
        "risk_from_others": "Risk from others / retaliation",
        "risk_of_self_neglect": "Risk of self-neglect",
        "risk_substance_use": "Risk from substance use"
    }
    for key, label in domain_labels.items():
        d = risk_data.get("domains", {}).get(key, {})
        if not d: continue
        level = d.get("level", "?")
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{label}**")
            if level == "LOW": c2.success(level)
            elif level == "MODERATE": c2.warning(level)
            else: c2.error(level)
            st.markdown(d.get("tribunal_paragraph", ""))
            if show_debug:
                with st.expander("Debug — evidence & scoring"):
                    st.markdown(f"**Past:** {d.get('past_factors','')}")
                    st.markdown(f"**Present:** {d.get('present_factors','')}")
                    st.markdown(f"**Future:** {d.get('future_factors','')}")
                    if d.get("increasing_factors"): st.markdown(f"**↑ Increasing:** {', '.join(d['increasing_factors'])}")
                    if d.get("reducing_factors"): st.markdown(f"**↓ Reducing:** {', '.join(d['reducing_factors'])}")
                    if d.get("evidence"):
                        st.markdown("**Evidence:**")
                        for e in d["evidence"]: st.markdown(f"- {e}")
    st.divider()
    st.markdown("**Overall risk summary**")
    st.markdown(risk_data.get("overall_risk_summary", ""))


def render_s3(s3_data, patient_name, show_debug):
    st.warning("⚠️ AI-assisted draft. Must be reviewed by the responsible clinician before use.")
    st.markdown("*I am approved under section 12 of the Act as having special experience in the diagnosis or treatment of mental disorder.*")
    fields = [
        ("Prior acquaintance with patient", "prior_acquaintance"),
        ("Nature of mental disorder", "nature_of_disorder"),
        ("Current symptoms", "current_symptoms"),
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
        val = s3_data.get(key) or "Not documented — clinician to complete"
        edits[key] = st.text_area(label, value=val, height=100, key=f"s3_{key}")
    if show_debug and s3_data.get("confidence_note"):
        with st.expander("AI confidence note"):
            st.info(s3_data["confidence_note"])
    st.divider()
    plain = f"MEDICAL RECOMMENDATION FOR ADMISSION FOR TREATMENT (SECTION 3 MHA 1983 — FORM A8)\nPatient: {patient_name}\n\n"
    plain += "I am approved under section 12 of the Act as having special experience in the diagnosis or treatment of mental disorder.\n\n"
    for label, key in fields:
        plain += f"{label.upper()}\n{edits.get(key,'')}\n\n"
    plain += "---\nAI-ASSISTED DRAFT — Must be reviewed and signed by an approved clinician.\nGenerated by PsySummarise (research prototype). Not validated for clinical use.\n"
    st.download_button("⬇️ Download Section 3 draft (.txt)", data=plain,
                       file_name=f"Section3_{patient_name}.txt", mime="text/plain")


def render_tribunal(tr_data, patient_name, tribunal_type_key, show_debug):
    st.warning("⚠️ AI-assisted draft. Must be reviewed by the Responsible Clinician before tribunal submission.")
    if tribunal_type_key == "Inpatient detention appeal":
        sections = [
            ("3. Factors affecting hearing", "q3_factors_affecting_hearing"),
            ("4. Adjustments", "q4_adjustments"),
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
            ("4. Adjustments", "q4_adjustments"),
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
    edits = {}
    for label, key in sections:
        val = tr_data.get(key) or "Not documented — clinician to complete"
        edits[key] = st.text_area(label, value=val, height=120, key=f"tr_{key}")
    if show_debug and tr_data.get("confidence_note"):
        with st.expander("AI confidence note"):
            st.info(tr_data["confidence_note"])
    st.divider()
    plain = f"RESPONSIBLE CLINICIAN'S REPORT FOR MENTAL HEALTH TRIBUNAL\n"
    plain += f"Tribunal type: {tr_data.get('tribunal_type','')}\nPatient: {patient_name}\nRC: {tr_data.get('rc_name','')}\n\n"
    for label, key in sections:
        plain += f"{label.upper()}\n{edits.get(key,'')}\n\n"
    plain += "---\nAI-ASSISTED DRAFT — Must be reviewed and signed by the Responsible Clinician.\nGenerated by PsySummarise (research prototype). Not validated for clinical use.\n"
    st.download_button("⬇️ Download tribunal report (.txt)", data=plain,
                       file_name=f"TribunalReport_{patient_name}.txt", mime="text/plain")


def note_input_widget(key_prefix):
    method = st.radio("Input method", ["Paste text", "Upload PDF", "Upload Word (.docx)"],
                      horizontal=True, key=f"{key_prefix}_method")
    if method == "Paste text":
        return st.text_area("Paste note here", height=180,
                            placeholder="Paste a ward round note, admission summary, MDT review, or discharge letter...",
                            key=f"{key_prefix}_paste")
    elif method == "Upload PDF":
        st.info("ℹ️ PDF must contain selectable text. Scanned/image-only PDFs cannot be read — please paste the text instead.")
        f = st.file_uploader("Upload PDF", type=["pdf"], key=f"{key_prefix}_pdf")
        if f:
            text, err = extract_text_from_file(f)
            if err == "pdf_empty":
                st.error("No text extracted. This may be a scanned image PDF. Please paste the text instead.")
            elif err:
                st.error(f"Could not read PDF: {err}")
            else:
                st.success(f"PDF read — {len(text.split())} words extracted.")
                return text
    elif method == "Upload Word (.docx)":
        f = st.file_uploader("Upload Word document", type=["docx"], key=f"{key_prefix}_docx")
        if f:
            text, err = extract_text_from_file(f)
            if err:
                st.error(f"Could not read document: {err}")
            else:
                st.success(f"Document read — {len(text.split())} words extracted.")
                return text
    return None


def add_notes_widget(key_prefix, session_notes_key, session_extracted_key, api_key):
    # ── Input method ──────────────────────────────────────────────────────────
    input_mode = st.radio("Input method", ["Paste text", "Upload files (PDF / Word)"],
                          horizontal=True, key=f"{key_prefix}_inputmode")

    if input_mode == "Paste text":
        note_text = st.text_area("Paste note here", height=160,
            placeholder="Paste a ward round note, admission summary, MDT review, or discharge letter...",
            key=f"{key_prefix}_paste")
        c1, c2 = st.columns([3, 1])
        with c1:
            add = st.button("Add note →", key=f"{key_prefix}_add")
        with c2:
            if st.button("Clear all", key=f"{key_prefix}_clear"):
                st.session_state[session_notes_key] = []
                st.session_state[session_extracted_key] = []
                st.rerun()
        if add:
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            elif not note_text or not note_text.strip():
                st.warning("Please provide a note above.")
            else:
                n = len(st.session_state[session_notes_key]) + 1
                with st.spinner(f"Extracting note {n}..."):
                    try:
                        extracted = extract_note(note_text, api_key)
                        st.session_state[session_notes_key].append(note_text)
                        st.session_state[session_extracted_key].append(extracted)
                        st.success(f"Note {n} added — {extracted.get('patient_id','?')} / {extracted.get('doc_id','?')}")
                    except Exception as e:
                        st.error(f"Extraction failed: {e}")

    else:
        st.info("ℹ️ PDFs must contain selectable text. Scanned/image-only PDFs cannot be read.")
        uploaded_files = st.file_uploader(
            "Upload one or more files (PDF or Word)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key=f"{key_prefix}_multiupload"
        )
        c1, c2 = st.columns([3, 1])
        with c1:
            add_files = st.button(f"Extract all {len(uploaded_files)} file(s) →" if uploaded_files else "Extract files →",
                                  key=f"{key_prefix}_addfiles",
                                  disabled=not uploaded_files)
        with c2:
            if st.button("Clear all", key=f"{key_prefix}_clear2"):
                st.session_state[session_notes_key] = []
                st.session_state[session_extracted_key] = []
                st.rerun()
        if add_files and uploaded_files:
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            else:
                progress = st.progress(0)
                for i, f in enumerate(uploaded_files):
                    text, err = extract_text_from_file(f)
                    if err == "pdf_empty":
                        st.error(f"{f.name}: No text extracted — may be a scanned PDF.")
                        continue
                    elif err:
                        st.error(f"{f.name}: Could not read — {err}")
                        continue
                    n = len(st.session_state[session_notes_key]) + 1
                    with st.spinner(f"Extracting {f.name}..."):
                        try:
                            extracted = extract_note(text, api_key)
                            st.session_state[session_notes_key].append(text)
                            st.session_state[session_extracted_key].append(extracted)
                            st.success(f"{f.name} → {extracted.get('patient_id','?')} / {extracted.get('doc_id','?')}")
                        except Exception as e:
                            st.error(f"{f.name}: Extraction failed — {e}")
                    progress.progress((i + 1) / len(uploaded_files))
                progress.empty()

    if st.session_state[session_extracted_key]:
        st.markdown(f"**{len(st.session_state[session_extracted_key])} note(s) loaded:**")
        for i, r in enumerate(st.session_state[session_extracted_key]):
            st.caption(f"{i+1}. {r.get('patient_id','?')} / {r.get('doc_id','?')} — "
                       f"{len(r.get('medications',[]))} med(s), {len(r.get('diagnoses',[]))} dx, "
                       f"{len(r.get('admissions',[]))} admission event(s)")


def stage_selector(key):
    options = [
        "Acute admission (first days)",
        "Mid-admission (settled but ongoing symptoms)",
        "Discharge-ready (nearing discharge)"
    ]
    stage = st.selectbox("Admission stage", options, index=1, key=f"{key}_stage")
    override = st.text_input("Override / add detail (optional)",
                             placeholder="e.g. 'Day 12, refusing antipsychotics, partial insight'",
                             key=f"{key}_override")
    return override.strip() if override.strip() else stage


# ── App layout ────────────────────────────────────────────────────────────────
st.title("🧠 PsySummarise")
st.caption("Structured extraction from psychiatric documentation · Research prototype · Not validated for clinical use")
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    st.divider()
    show_debug = st.toggle("Debug mode", value=False,
                           help="Shows evidence citations, risk scoring, and AI confidence notes")
    st.divider()
    st.caption("PsySummarise uses GPT-4o. Research prototype. Not validated for clinical use.")

for key in ["s3_notes","s3_extracted","tr_notes","tr_extracted"]:
    if key not in st.session_state:
        st.session_state[key] = []

tab1, tab2, tab3 = st.tabs(["📄 Single note extraction", "📋 Section 3 recommendation", "⚖️ Tribunal report"])

# Tab 1
with tab1:
    note_text = note_input_widget("t1")
    if st.button("Extract structured data →", key="extract_single"):
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        elif not note_text or not note_text.strip():
            st.warning("Please provide a note above.")
        else:
            with st.spinner("Extracting..."):
                try:
                    result = extract_note(note_text, api_key)
                    st.success("Extraction complete.")
                    st.divider()
                    s_tab, j_tab = st.tabs(["Clinical summary", "Raw JSON"])
                    with s_tab:
                        render_clinical_summary(result)
                    with j_tab:
                        st.json(result)
                        st.download_button("⬇️ Download JSON",
                            data=json.dumps(result, indent=2, ensure_ascii=False),
                            file_name=f"psysummarise_{result.get('patient_id','output')}.json",
                            mime="application/json")
                except Exception as e:
                    st.error(f"Error: {e}")

# Tab 2
with tab2:
    st.markdown("### Build a Section 3 recommendation from multiple notes")
    st.caption("Add each clinical document one at a time. Select admission stage before generating.")
    add_notes_widget("s3", "s3_notes", "s3_extracted", api_key)
    if st.session_state.s3_extracted:
        st.divider()
        stage = stage_selector("s3")
        if st.button("Generate Section 3 recommendation →", key="gen_s3"):
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            else:
                risk_result = {}
                with st.spinner("Computing risk assessment..."):
                    try:
                        risk_result = compute_risk(st.session_state.s3_extracted, st.session_state.s3_notes, stage, api_key)
                    except Exception as e:
                        st.warning(f"Risk assessment failed: {e}")
                with st.spinner("Generating Section 3 recommendation..."):
                    try:
                        s3_result = generate_s3(st.session_state.s3_extracted, st.session_state.s3_notes, risk_result, api_key)
                        st.success("Draft generated. Review all fields carefully.")
                        st.divider()
                        patient_name = s3_result.get("patient_name") or \
                            (st.session_state.s3_extracted[0].get("patient_id","Patient"))
                        r_tab, d_tab, src_tab = st.tabs(["Risk assessment", "Draft recommendation", "Source data"])
                        with r_tab:
                            render_risk(risk_result, show_debug, len(st.session_state.s3_extracted))
                        with d_tab:
                            render_s3(s3_result, patient_name, show_debug)
                        with src_tab:
                            st.json(st.session_state.s3_extracted)
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.info("Add at least one note above to begin.")

# Tab 3
with tab3:
    st.markdown("### Generate a tribunal report from multiple notes")
    st.caption("Add each clinical document one at a time. Select tribunal type and admission stage before generating.")
    tribunal_type = st.radio("Tribunal type",
        ["Inpatient detention appeal (Section 2 / Section 3)", "CTO appeal"], horizontal=True)
    tribunal_type_key = "Inpatient detention appeal" if "Inpatient" in tribunal_type else "CTO appeal"
    add_notes_widget("tr", "tr_notes", "tr_extracted", api_key)
    if st.session_state.tr_extracted:
        st.divider()
        stage = stage_selector("tr")
        if st.button("Generate tribunal report →", key="gen_tribunal"):
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            else:
                tr_risk = {}
                with st.spinner("Computing risk assessment..."):
                    try:
                        tr_risk = compute_risk(st.session_state.tr_extracted, st.session_state.tr_notes, stage, api_key)
                    except Exception as e:
                        st.warning(f"Risk assessment failed: {e}")
                with st.spinner("Generating tribunal report — this may take 30–60 seconds..."):
                    try:
                        tr_result = generate_tribunal(
                            st.session_state.tr_extracted, st.session_state.tr_notes,
                            tr_risk, tribunal_type_key, stage, api_key
                        )
                        st.success("Draft generated. Review all sections carefully.")
                        st.divider()
                        patient_name = tr_result.get("patient_name") or \
                            (st.session_state.tr_extracted[0].get("patient_id","Patient"))
                        r_tab, d_tab, src_tab = st.tabs(["Risk assessment", "Draft report", "Source data"])
                        with r_tab:
                            render_risk(tr_risk, show_debug, len(st.session_state.tr_extracted))
                        with d_tab:
                            render_tribunal(tr_result, patient_name, tribunal_type_key, show_debug)
                        with src_tab:
                            st.json(st.session_state.tr_extracted)
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.info("Add at least one note above to begin.")
