import streamlit as st
import json
import os
from openai import OpenAI

st.set_page_config(
    page_title="PsySummarise",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
.stTextArea textarea {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    background: #1a1d27;
    border: 1px solid #2a2d3e;
    color: #e2e8f0;
    border-radius: 8px;
}
.stButton > button {
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    padding: 0.6rem 2rem;
    width: 100%;
    font-size: 14px;
    letter-spacing: 0.05em;
}
.stButton > button:hover { background: #2563eb; }
.field-card {
    background: #1a1d27;
    border: 1px solid #2a2d3e;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}
.field-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.field-value { font-size: 15px; color: #e2e8f0; }
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #3b82f6;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid #2a2d3e;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem;
}
.badge { display:inline-block;padding:2px 10px;border-radius:99px;font-size:12px;font-family:'IBM Plex Mono',monospace;font-weight:500;margin-right:6px; }
.badge-blue  { background:#1e3a5f;color:#60a5fa; }
.badge-green { background:#14291f;color:#34d399; }
.badge-amber { background:#2d1f06;color:#fbbf24; }
.badge-red   { background:#2d0f0f;color:#f87171; }
.badge-gray  { background:#1f2130;color:#9ca3af; }
.mono-output {
    font-family:'IBM Plex Mono',monospace;font-size:12px;
    background:#1a1d27;border:1px solid #2a2d3e;border-radius:8px;
    padding:1rem;color:#a5f3fc;white-space:pre-wrap;
    max-height:600px;overflow-y:auto;
}
.s3-draft {
    background:#1a1d27;border-left:3px solid #3b82f6;
    border-radius:0 10px 10px 0;padding:1.5rem;
    color:#cbd5e1;font-size:14px;line-height:1.8;
    white-space:pre-wrap;font-family:'IBM Plex Sans',sans-serif;
}
.warning-box {
    background:#2d1f06;border:1px solid #92400e;border-radius:8px;
    padding:0.8rem 1rem;color:#fbbf24;font-size:13px;margin:1rem 0;
}
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
draft a Section 3 MHA recommendation.

Extracted clinical data:
{extracted_data}

Return a JSON object with exactly these fields:
{{
  "patient_name": "First name of patient",
  "nature_of_disorder": "2-3 sentence description of the mental disorder, its nature and current presentation",
  "current_symptoms": "Key current symptoms and mental state findings",
  "risk_to_self": "Current risk to self with supporting evidence",
  "risk_to_others": "Current risk to others with supporting evidence",
  "why_informal_insufficient": "Clinical reasoning for why informal admission is not appropriate",
  "why_section2_insufficient": "Clinical reasoning for why Section 2 is no longer sufficient (omit if not applicable)",
  "treatment_available": "What treatment is available and appropriate in hospital",
  "why_community_insufficient": "Why treatment cannot be provided without detention",
  "medication_history": "Brief summary of medication history and compliance",
  "recommendation": "A clear concluding statement recommending Section 3 detention",
  "confidence_note": "Any fields where information was limited or uncertain — be explicit here"
}}

Only use information present in the extracted data. Where information is absent, state 'Not documented in available notes' rather than inferring."""


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


def adherence_badge(val):
    colours = {"good":"green","partial":"amber","poor":"red","unknown":"gray"}
    c = colours.get(str(val).lower(),"gray")
    return f'<span class="badge badge-{c}">{val or "unknown"}</span>'

def certainty_badge(val):
    c = "green" if val == "clear" else "amber"
    return f'<span class="badge badge-{c}">{val or "?"}</span>'

def legal_badge(val):
    return f'<span class="badge badge-blue">{val}</span>' if val else ""


def render_clinical_summary(data):
    patient = data.get("patient_id","Unknown")
    doc = data.get("doc_id","Unknown")
    st.markdown(f"""
    <div style="display:flex;align-items:baseline;gap:1rem;margin-bottom:1.5rem">
        <h2 style="margin:0;color:#e2e8f0;font-size:22px">{patient}</h2>
        <span style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#6b7280">{doc}</span>
    </div>""", unsafe_allow_html=True)

    admissions = data.get("admissions",[])
    if admissions:
        st.markdown('<div class="section-header">Admissions & Legal Status</div>', unsafe_allow_html=True)
        for adm in admissions:
            ls = adm.get("legal_status")
            event_type = adm.get("event_type","?")
            date = adm.get("event_date") or "date unknown"
            reason = adm.get("reason")
            quote = adm.get("evidence_quote")
            st.markdown(f"""<div class="field-card">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
                    {legal_badge(ls)}
                    <span class="badge badge-gray">{event_type}</span>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:#6b7280">{date}</span>
                </div>
                {"<div class='field-label'>Reason</div><div class='field-value' style='font-size:13px'>" + reason + "</div>" if reason else ""}
                {"<div class='field-label' style='margin-top:8px;color:#4b5563'>Evidence</div><div style='font-size:12px;color:#6b7280;font-style:italic'>&ldquo;" + quote + "&rdquo;</div>" if quote else ""}
            </div>""", unsafe_allow_html=True)

    diagnoses = data.get("diagnoses",[])
    if diagnoses:
        st.markdown('<div class="section-header">Diagnoses</div>', unsafe_allow_html=True)
        for dx in diagnoses:
            label = dx.get("dx_label") or "Unknown"
            icd = dx.get("icd10")
            status = dx.get("status")
            mdt = dx.get("mdt_agreement")
            certainty = dx.get("certainty")
            quote = dx.get("evidence_quote")
            st.markdown(f"""<div class="field-card">
                <div class="field-value" style="font-size:16px;font-weight:500;margin-bottom:8px">{label}</div>
                <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px">
                    {"<span class='badge badge-blue'>" + icd + "</span>" if icd else ""}
                    {"<span class='badge badge-gray'>" + status + "</span>" if status else ""}
                    {"<span class='badge badge-" + ("green" if mdt=="agree" else "amber") + "'>MDT: " + str(mdt) + "</span>" if mdt else ""}
                    {certainty_badge(certainty)}
                </div>
                {"<div class='field-label' style='color:#4b5563'>Evidence</div><div style='font-size:12px;color:#6b7280;font-style:italic'>&ldquo;" + quote + "&rdquo;</div>" if quote else ""}
            </div>""", unsafe_allow_html=True)

    medications = data.get("medications",[])
    if medications:
        st.markdown('<div class="section-header">Medications</div>', unsafe_allow_html=True)
        for med in medications:
            action = med.get("action","?")
            name = med.get("med_name") or "Unknown"
            dose = med.get("dose_text")
            route = med.get("route")
            adherence = med.get("adherence","unknown")
            response = med.get("response")
            side_effects = med.get("side_effects") or []
            reason = med.get("reason_change")
            certainty = med.get("certainty")
            quote = med.get("evidence_quote")
            ac = {"start":"green","continue":"blue","change":"amber","stop":"red"}.get(action,"gray")
            st.markdown(f"""<div class="field-card">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
                    <span class="badge badge-{ac}">{action}</span>
                    <span style="font-size:16px;font-weight:500;color:#e2e8f0">{name}</span>
                    {"<span style='font-family:IBM Plex Mono,monospace;font-size:13px;color:#9ca3af'>" + dose + "</span>" if dose else ""}
                </div>
                <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px">
                    {"<span class='badge badge-gray'>" + route + "</span>" if route else ""}
                    {adherence_badge(adherence)}
                    {certainty_badge(certainty)}
                </div>
                {"<div class='field-label'>Response</div><div class='field-value' style='font-size:13px'>" + response + "</div>" if response else ""}
                {"<div class='field-label' style='margin-top:8px'>Side effects</div><div class='field-value' style='font-size:13px'>" + ", ".join(side_effects) + "</div>" if side_effects else ""}
                {"<div class='field-label' style='margin-top:8px'>Reason for change</div><div class='field-value' style='font-size:13px'>" + reason + "</div>" if reason else ""}
                {"<div class='field-label' style='margin-top:8px;color:#4b5563'>Evidence</div><div style='font-size:12px;color:#6b7280;font-style:italic'>&ldquo;" + quote + "&rdquo;</div>" if quote else ""}
            </div>""", unsafe_allow_html=True)


def render_s3(s3_data, patient_name):
    st.markdown('<div class="warning-box">⚠ This is an AI-assisted draft. All content must be reviewed, verified, and approved by the responsible clinician before use. Not for direct clinical submission.</div>', unsafe_allow_html=True)

    fields = [
        ("Nature of disorder", "nature_of_disorder"),
        ("Current symptoms", "current_symptoms"),
        ("Risk to self", "risk_to_self"),
        ("Risk to others", "risk_to_others"),
        ("Why informal admission is insufficient", "why_informal_insufficient"),
        ("Why Section 2 is insufficient", "why_section2_insufficient"),
        ("Available treatment in hospital", "treatment_available"),
        ("Why community treatment is insufficient", "why_community_insufficient"),
        ("Medication history and compliance", "medication_history"),
        ("Recommendation", "recommendation"),
    ]

    edits = {}
    for label, key in fields:
        val = s3_data.get(key, "Not documented in available notes")
        if not val or val.strip() == "":
            val = "Not documented in available notes"
        st.markdown(f'<div class="field-label" style="margin-top:1rem">{label}</div>', unsafe_allow_html=True)
        edits[key] = st.text_area(
            label=label,
            value=val,
            height=100,
            key=f"s3_{key}",
            label_visibility="collapsed"
        )

    confidence = s3_data.get("confidence_note","")
    if confidence:
        st.markdown(f"""<div style="background:#1a1d27;border:1px solid #374151;border-radius:8px;padding:1rem;margin-top:1rem">
            <div class="field-label" style="color:#6b7280">Confidence note (AI self-assessment)</div>
            <div style="font-size:13px;color:#9ca3af;margin-top:4px">{confidence}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Export as plain text**")

    plain = f"SECTION 3 MHA RECOMMENDATION\nPatient: {patient_name}\n\n"
    for label, key in fields:
        plain += f"{label.upper()}\n{edits.get(key,'')}\n\n"

    st.download_button(
        "Download Section 3 draft (.txt)",
        data=plain,
        file_name=f"Section3_{patient_name.replace(' ','_')}.txt",
        mime="text/plain"
    )


# ── Page header ──────────────────────────────────────────────────────────────

col1, col2 = st.columns([1,3])
with col1:
    st.markdown("# 🧠")
with col2:
    st.markdown("# PsySummarise")
    st.markdown('<p style="color:#6b7280;font-family:\'IBM Plex Mono\',monospace;font-size:13px;margin-top:-10px">Structured extraction from psychiatric documentation</p>', unsafe_allow_html=True)

st.divider()

with st.sidebar:
    st.markdown("### Settings")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    st.markdown("---")
    st.markdown('<p style="font-size:12px;color:#6b7280">PsySummarise extracts structured clinical information from psychiatric ward round notes using GPT-4o.<br><br>Built as a research prototype. Not validated for clinical use.</p>', unsafe_allow_html=True)

# ── Mode tabs ─────────────────────────────────────────────────────────────────

mode = st.tabs(["Single note extraction", "Section 3 recommendation"])

# ── Tab 1: Single note ────────────────────────────────────────────────────────
with mode[0]:
    note_text = st.text_area(
        "Paste ward round note",
        height=280,
        placeholder="Paste a psychiatric ward round note, admission summary, or discharge letter here..."
    )
    run = st.button("Extract structured data →", key="extract_single")

    if run:
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
                    tab1, tab2 = st.tabs(["Clinical summary", "Raw JSON"])
                    with tab1:
                        render_clinical_summary(result)
                    with tab2:
                        st.markdown(f'<div class="mono-output">{json.dumps(result, indent=2, ensure_ascii=False)}</div>', unsafe_allow_html=True)
                        st.download_button(
                            "Download JSON",
                            data=json.dumps(result, indent=2, ensure_ascii=False),
                            file_name=f"psysummarise_{result.get('patient_id','output')}.json",
                            mime="application/json"
                        )
                except json.JSONDecodeError as e:
                    st.error(f"Model returned invalid JSON: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Tab 2: Section 3 ──────────────────────────────────────────────────────────
with mode[1]:
    st.markdown("### Build a Section 3 recommendation from multiple notes")
    st.markdown('<p style="color:#6b7280;font-size:13px">Add each ward round note or clinical document one at a time. The system will extract from each and then synthesise a draft Section 3 recommendation across all notes.</p>', unsafe_allow_html=True)

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

    col_add, col_clear = st.columns([3,1])
    with col_add:
        if st.button("Add note →", key="add_note"):
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            elif not new_note.strip():
                st.warning("Please paste a note above.")
            else:
                with st.spinner(f"Extracting note {len(st.session_state.s3_notes)+1}..."):
                    try:
                        extracted = extract_note(new_note, api_key)
                        st.session_state.s3_notes.append(new_note)
                        st.session_state.s3_extracted.append(extracted)
                        st.success(f"Note {len(st.session_state.s3_notes)} added — {extracted.get('patient_id','?')} / {extracted.get('doc_id','?')}")
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
            pid = rec.get("patient_id","?")
            did = rec.get("doc_id","?")
            n_meds = len(rec.get("medications",[]))
            n_dx = len(rec.get("diagnoses",[]))
            n_adm = len(rec.get("admissions",[]))
            st.markdown(
                f'<div style="font-size:13px;font-family:IBM Plex Mono,monospace;color:#9ca3af;padding:4px 0;border-bottom:1px solid #2a2d3e">'
                f'{i+1}. {pid} / {did} — {n_meds} med(s), {n_dx} dx, {n_adm} admission event(s)</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
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
                            (st.session_state.s3_extracted[0].get("patient_id","Patient") if st.session_state.s3_extracted else "Patient")

                        s3_tab1, s3_tab2 = st.tabs(["Draft recommendation", "Source JSON"])

                        with s3_tab1:
                            render_s3(s3_result, patient_name)

                        with s3_tab2:
                            st.markdown("**Extracted data used to generate this recommendation:**")
                            st.markdown(
                                f'<div class="mono-output">{json.dumps(st.session_state.s3_extracted, indent=2, ensure_ascii=False)}</div>',
                                unsafe_allow_html=True
                            )

                    except json.JSONDecodeError as e:
                        st.error(f"Model returned invalid JSON: {e}")
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.info("Add at least one note above to begin.")
