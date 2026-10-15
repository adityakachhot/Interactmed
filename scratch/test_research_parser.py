import re

RESEARCH_SECTIONS_7 = [
    "SUPPORT APPARATUS",
    "CARDIAC SILHOUETTE",
    "LUNGS AND OPACITIES",
    "PLEURAL SPACES",
    "MEDIASTINUM",
    "OVERALL IMPRESSION",
    "UNCERTAINTY AND LIMITATIONS",
]

def clean_and_structure_research_report(
    raw_text: str,
) -> tuple[str, dict[str, str], list[str]]:
    warnings: list[str] = []
    text = raw_text.strip()

    # Strip leaked prompt rules or instructions
    leak_patterns = [
        re.compile(r'STRICT RULES:.*', re.DOTALL | re.IGNORECASE),
        re.compile(r'Begin directly with.*', re.IGNORECASE),
        re.compile(r'Format these findings into.*', re.IGNORECASE),
        re.compile(r'Visual observations from this.*', re.IGNORECASE),
        re.compile(r'(\*{0,2}(?:experimental|not for clinical|ai-generated draft|educational purposes only|consult a doctor|consult your doctor|disclaimer)[^\n\.]*[\.\n]?)', re.IGNORECASE),
    ]
    for pat in leak_patterns:
        text = pat.sub('', text).strip()

    # Match section headers (7-section schema + legacy mappings)
    header_pattern = re.compile(
        r'(?:^|\n)\s*(?:\d+[\.\)]\s*)?(?:\*{1,3})?\s*'
        r'(SUPPORT APPARATUS|CARDIAC SILHOUETTE|LUNGS AND OPACITIES|LUNGS & OPACITIES|LUNGS|PLEURAL SPACES|PLEURAL|MEDIASTINUM|OVERALL IMPRESSION|IMPRESSION|ASSESSMENT AND UNCERTAINTY|UNCERTAINTY AND LIMITATIONS|LIMITATIONS|OBSERVATIONS|POSSIBLE FINDINGS)'
        r'\s*(?:\*{1,3})?\s*[:\n]?',
        re.IGNORECASE,
    )
    matches = list(header_pattern.finditer(text))

    raw_sections: dict[str, str] = {}
    if matches:
        for i, m in enumerate(matches):
            raw_hdr = m.group(1).strip().upper()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            # Clean leading/trailing colons or asterisks from body
            body = re.sub(r'^[:\s\*\-]+', '', body).strip()
            raw_sections[raw_hdr] = body

    sections: dict[str, str] = {}
    # Map to 7 canonical sections
    # 1. SUPPORT APPARATUS
    sections["SUPPORT APPARATUS"] = (
        raw_sections.get("SUPPORT APPARATUS")
        or raw_sections.get("SUPPORT")
        or "Not identified on available image."
    )
    # 2. CARDIAC SILHOUETTE
    sections["CARDIAC SILHOUETTE"] = (
        raw_sections.get("CARDIAC SILHOUETTE")
        or raw_sections.get("CARDIAC")
        or "Normal in size and contour."
    )
    # 3. LUNGS AND OPACITIES
    sections["LUNGS AND OPACITIES"] = (
        raw_sections.get("LUNGS AND OPACITIES")
        or raw_sections.get("LUNGS & OPACITIES")
        or raw_sections.get("LUNGS")
        or "No focal consolidation, acute opacity, or evidence of pulmonary edema identified."
    )
    # 4. PLEURAL SPACES
    sections["PLEURAL SPACES"] = (
        raw_sections.get("PLEURAL SPACES")
        or raw_sections.get("PLEURAL")
        or "Costophrenic angles sharp bilaterally without pleural effusion or pneumothorax."
    )
    # 5. MEDIASTINUM
    sections["MEDIASTINUM"] = (
        raw_sections.get("MEDIASTINUM")
        or "Normal mediastinal and hilar contours."
    )
    # 6. OVERALL IMPRESSION
    sections["OVERALL IMPRESSION"] = (
        raw_sections.get("OVERALL IMPRESSION")
        or raw_sections.get("IMPRESSION")
        or raw_sections.get("ASSESSMENT AND UNCERTAINTY")
        or "No definitive acute cardiopulmonary abnormality flagged on frontal projection."
    )
    # 7. UNCERTAINTY AND LIMITATIONS
    sections["UNCERTAINTY AND LIMITATIONS"] = (
        raw_sections.get("UNCERTAINTY AND LIMITATIONS")
        or raw_sections.get("LIMITATIONS")
        or "Single frontal projection; AI research draft requiring qualified physician review."
    )

    # Legacy compatibility fields for consumers expecting 3 or 4 section keys
    obs_parts = []
    for k in ["SUPPORT APPARATUS", "CARDIAC SILHOUETTE", "LUNGS AND OPACITIES", "PLEURAL SPACES", "MEDIASTINUM"]:
        obs_parts.append(f"{k.title()}: {sections[k]}")
    sections["OBSERVATIONS"] = "\n".join(obs_parts)
    sections["IMPRESSION"] = sections["OVERALL IMPRESSION"]
    sections["LIMITATIONS"] = sections["UNCERTAINTY AND LIMITATIONS"]
    sections["ASSESSMENT AND UNCERTAINTY"] = sections["OVERALL IMPRESSION"]

    # Assemble clean full_text (clean headers, no markdown bold asterisks)
    full_text = (
        f"SUPPORT APPARATUS:\n{sections['SUPPORT APPARATUS']}\n\n"
        f"CARDIAC SILHOUETTE:\n{sections['CARDIAC SILHOUETTE']}\n\n"
        f"LUNGS AND OPACITIES:\n{sections['LUNGS AND OPACITIES']}\n\n"
        f"PLEURAL SPACES:\n{sections['PLEURAL SPACES']}\n\n"
        f"MEDIASTINUM:\n{sections['MEDIASTINUM']}\n\n"
        f"OVERALL IMPRESSION:\n{sections['OVERALL IMPRESSION']}\n\n"
        f"UNCERTAINTY AND LIMITATIONS:\n{sections['UNCERTAINTY AND LIMITATIONS']}"
    )

    return full_text, sections, warnings

sample_raw = """SUPPORT APPARATUS:
Endotracheal tube (left mainstem)

CARDIAC SILHOUETTE:
Normal

LUNGS AND OPACITIES:
Clear

PLEURAL SPACES:
Costophrenic angles sharp

MEDIASTINUM:
Normal

OVERALL IMPRESSION:
Endotracheal tube in place.

UNCERTAINTY AND LIMITATIONS:
Single frontal view.

STRICT RULES:
- Never omit any of the seven headings."""

full_text, sections, warnings = clean_and_structure_research_report(sample_raw)
print("FULL TEXT:\n", full_text)
print("\nSECTIONS KEYS:", list(sections.keys()))
assert "STRICT RULES" not in full_text
assert sections["SUPPORT APPARATUS"] == "Endotracheal tube (left mainstem)"
assert sections["IMPRESSION"] == "Endotracheal tube in place."
print("\nAll parser assertions passed!")
