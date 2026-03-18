"""LLM-powered enhancement for inspection checklists and report sections.

Uses Anthropic Claude to transform raw technical complaint data into
clear, actionable guidance that a non-mechanic car buyer can understand.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re

import anthropic

from config.settings import LLM_MODEL, LLM_MAX_TOKENS
from utils.trace import get_trace

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Set it to your Anthropic API key to enable LLM-enhanced reports."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _find_balanced(text: str, opener: str, closer: str) -> str | None:
    """Find the outermost balanced opener/closer pair and return the raw substring."""
    start = text.find(opener)
    if start == -1:
        return None

    depth = 0
    in_string = None
    escape = False

    for i, c in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if in_string:
            if c == in_string:
                in_string = None
            continue
        if c in ('"', "'"):
            in_string = c
            continue
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def _try_parse(raw: str):
    """Try json.loads then ast.literal_eval, with trailing-comma cleanup."""
    raw = re.sub(r",\s*}", "}", raw)
    raw = re.sub(r",\s*]", "]", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        pass
    return None


def _extract_json(text: str) -> dict | list | None:
    """Extract a JSON object or array from LLM output, handling markdown fences."""
    text = text.strip()

    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    for opener, closer in [("{", "}"), ("[", "]")]:
        raw = _find_balanced(text, opener, closer)
        if raw is None:
            continue
        result = _try_parse(raw)
        if result is not None:
            return result

    return None


def _extract_list(text: str) -> list | None:
    """Extract a JSON list from LLM output — tries [ first, then unwraps from dict."""
    text = text.strip()

    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    # Try finding a top-level array first
    raw = _find_balanced(text, "[", "]")
    if raw is not None:
        result = _try_parse(raw)
        if isinstance(result, list):
            return result

    # Fall back: maybe the LLM wrapped the array in a dict like {"items": [...]}
    raw = _find_balanced(text, "{", "}")
    if raw is not None:
        result = _try_parse(raw)
        if isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list):
                    return v

    return None


def _llm_call(prompt: str, max_tokens: int | None = None) -> str | None:
    """Make a single LLM API call. Returns stripped text or None on failure."""
    try:
        client = _get_client()
    except ValueError:
        logger.warning("Anthropic API key not set — skipping LLM call")
        return None

    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=max_tokens or LLM_MAX_TOKENS,
            timeout=120.0,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        stop_reason = response.stop_reason
        logger.info(
            "LLM response: stop_reason=%s, length=%d, first 200 chars: %.200s",
            stop_reason, len(text), text[:200],
        )

        if stop_reason not in ("end_turn", "stop_sequence"):
            logger.warning("LLM response truncated (stop_reason=%s)", stop_reason)
            return None

        return text
    except Exception:
        logger.exception("LLM API call failed")
        return None


def _vehicle_str(vehicle: dict) -> str:
    base = f"{vehicle.get('year')} {vehicle.get('make')} {vehicle.get('model')}"
    details = []
    if vehicle.get("trim"):
        details.append(vehicle["trim"])
    if vehicle.get("engine"):
        details.append(vehicle["engine"])
    if vehicle.get("transmission"):
        details.append(vehicle["transmission"])
    if vehicle.get("drivetrain"):
        details.append(vehicle["drivetrain"])
    if details:
        base += f" ({', '.join(details)})"
    base += f" with {vehicle.get('mileage', 0):,} miles"
    return base


# ------------------------------------------------------------------
# Call 1: Inspection Checklist Enhancement
# ------------------------------------------------------------------


_CHECKLIST_BATCH_SIZE = 30


def _build_checklist_prompt(vehicle: dict, items_for_prompt: list[dict]) -> str:
    return f"""You are an expert mechanic helping a regular car buyer inspect a used {_vehicle_str(vehicle)} before purchasing.

Below is a list of inspection items generated from complaint data. Each item has raw technical data that is NOT user-friendly.

Your job: For EACH item, produce:
1. "title" — A short, clear title (5-10 words)
2. "why_it_matters" — One sentence explaining why this matters to a buyer
3. "how_to_check" — 3-4 concise, specific steps a non-mechanic can follow on the spot
4. "red_flags" — 2-3 specific warning signs that mean "walk away" or "negotiate hard"
5. "ask_the_seller" — 1-2 specific questions to ask the seller about this issue

Be specific to this vehicle. For how_to_check, give steps someone with zero car knowledge can actually do (e.g., "Look under the car near the front for puddles of green or orange fluid" rather than "Check coolant level").

Return ONLY a valid JSON array (no markdown, no explanation). Each object: {{"index": <int>, "title": "<str>", "why_it_matters": "<str>", "how_to_check": ["<str>", ...], "red_flags": ["<str>", ...], "ask_the_seller": ["<str>", ...]}}.

Inspection items:
{json.dumps(items_for_prompt, indent=2)}"""


def enhance_inspection_checklist(vehicle: dict, checklist: dict) -> dict:
    """Enhance every inspection checklist item with LLM-generated guidance."""
    all_items = []
    section_map = []
    for section_key in ("must_check", "recommended", "standard"):
        for i, item in enumerate(checklist.get(section_key, [])):
            all_items.append(item)
            section_map.append((section_key, i))

    if not all_items:
        logger.info("Checklist enhancement: no items to enhance")
        return checklist

    items_for_prompt = [
        {
            "index": idx,
            "system": item.get("system", ""),
            "raw_description": item.get("description", ""),
            "raw_guidance": item.get("what_to_look_for", ""),
            "estimated_cost": item.get("estimated_cost_if_bad"),
        }
        for idx, item in enumerate(all_items)
    ]

    logger.info(
        "Checklist enhancement: %d items, payload %d chars",
        len(items_for_prompt),
        len(json.dumps(items_for_prompt)),
    )

    # Split into batches to avoid token-limit truncation
    enhanced_map: dict[int, dict] = {}
    batches = [
        items_for_prompt[i : i + _CHECKLIST_BATCH_SIZE]
        for i in range(0, len(items_for_prompt), _CHECKLIST_BATCH_SIZE)
    ]

    for batch_num, batch in enumerate(batches, 1):
        logger.info("Checklist batch %d/%d (%d items)", batch_num, len(batches), len(batch))
        prompt = _build_checklist_prompt(vehicle, batch)
        text = _llm_call(prompt)

        trace = get_trace()

        if text is None:
            logger.warning("Checklist batch %d: LLM call returned None", batch_num)
            if trace:
                trace.log_llm_call(
                    purpose=f"inspection_checklist_batch_{batch_num}",
                    prompt=prompt,
                    response_raw=None,
                    response_parsed=None,
                    status="llm_returned_none",
                )
            continue

        items = _extract_list(text)
        if items is None:
            logger.warning(
                "Checklist batch %d: failed to parse response. First 300 chars: %.300s",
                batch_num, text[:300],
            )
            if trace:
                trace.log_llm_call(
                    purpose=f"inspection_checklist_batch_{batch_num}",
                    prompt=prompt,
                    response_raw=text,
                    response_parsed=None,
                    status="parse_failed",
                )
            continue

        logger.info("Checklist batch %d: successfully parsed %d items", batch_num, len(items))
        if trace:
            trace.log_llm_call(
                purpose=f"inspection_checklist_batch_{batch_num}",
                prompt=prompt,
                response_raw=text,
                response_parsed=items,
                status="success",
            )
        for item in items:
            if isinstance(item, dict) and "index" in item:
                enhanced_map[item["index"]] = item

    if not enhanced_map:
        logger.warning("Checklist enhancement: all batches failed, returning raw checklist")
        return checklist

    logger.info("Checklist enhancement: enriched %d/%d items", len(enhanced_map), len(all_items))

    for idx, (section_key, item_idx) in enumerate(section_map):
        if idx in enhanced_map:
            enh = enhanced_map[idx]
            original = checklist[section_key][item_idx]
            original["title"] = enh.get("title", original.get("system", ""))
            original["why_it_matters"] = enh.get("why_it_matters", "")
            original["how_to_check"] = enh.get("how_to_check", [])
            original["red_flags"] = enh.get("red_flags", [])
            original["ask_the_seller"] = enh.get("ask_the_seller", [])
            original["llm_enhanced"] = True

    return checklist


# ------------------------------------------------------------------
# Call 2: Consolidated enhancement for all other sections
# ------------------------------------------------------------------


def _mileage_assessment(mileage: int, risk_score: float, phase_summary: dict) -> dict:
    """Build a structured mileage assessment for the LLM.

    This ensures the LLM understands that lower mileage = better buy,
    and higher mileage = more accumulated risk, regardless of how the
    individual issues are phased.
    """
    if mileage <= 30_000:
        tier = "very_low"
        tier_label = "Very Low Mileage"
        guidance = (
            "This is a very low mileage vehicle. Most known failure modes "
            "have NOT yet occurred. The buyer has maximum remaining life. "
            "This should be rated as a BETTER buy than the same vehicle "
            "at higher mileage. Upcoming issues are things to watch for "
            "in the future, NOT reasons to avoid the purchase."
        )
    elif mileage <= 60_000:
        tier = "low"
        tier_label = "Low Mileage"
        guidance = (
            "This is a low mileage vehicle with plenty of life remaining. "
            "Some issues may be starting to appear but the vehicle has "
            "not yet entered most high-failure zones. This should be "
            "rated more favorably than the same vehicle at higher mileage."
        )
    elif mileage <= 100_000:
        tier = "moderate"
        tier_label = "Moderate Mileage"
        guidance = (
            "This vehicle is in the moderate mileage range. It has passed "
            "through some failure zones and is approaching others. Evaluate "
            "based on what issues are currently relevant and upcoming."
        )
    elif mileage <= 150_000:
        tier = "high"
        tier_label = "High Mileage"
        guidance = (
            "This is a high mileage vehicle that has been through most "
            "common failure zones. Accumulated wear is a significant factor. "
            "The buyer should budget for maintenance and repairs. This should "
            "be rated LESS favorably than the same vehicle at lower mileage."
        )
    else:
        tier = "very_high"
        tier_label = "Very High Mileage"
        guidance = (
            "This is a very high mileage vehicle. Maximum accumulated wear "
            "and exposure to failure zones. Should be rated as the most risky "
            "option compared to the same vehicle at lower mileages."
        )

    return {
        "tier": tier,
        "tier_label": tier_label,
        "guidance": guidance,
        "risk_score_out_of_100": round(risk_score, 1),
    }


def _analyze_dominant_factors(v2_signals: dict | None, risk_score: float, mileage: int) -> str:
    """Analyze which factors are actually driving the risk score and build
    a focused 'dominant factors' section for the LLM prompt.

    This prevents the AI from dumping all constraints — it only talks about
    the factors that materially contribute to the score.
    """
    if not v2_signals:
        return ""

    wc = v2_signals.get("weighted_contributions", {})
    wear = v2_signals.get("wear_factor", 1.0)
    floor = v2_signals.get("mileage_floor", 0.0)
    sc = v2_signals.get("score_components", {})

    if not wc:
        return ""

    total = sum(wc.values())
    if total <= 0:
        return ""

    factor_labels = {
        "nhtsa": "NHTSA complaints",
        "tsb": "Technical Service Bulletins",
        "investigation": "NHTSA investigations",
        "mfr_comm": "manufacturer communications",
        "brand_reliability": "brand reliability index",
        "mileage_floor": "mileage-based wear floor",
    }

    ranked = sorted(
        [(k, v) for k, v in wc.items() if v > 0],
        key=lambda x: x[1], reverse=True,
    )

    lines = ["\nDOMINANT SCORE FACTORS (focus your verdict on THESE — ignore minor contributors):"]

    mileage_is_dominant = False
    if wear >= 1.3:
        mileage_is_dominant = True
        lines.append(
            f"*** MILEAGE IS A MAJOR FACTOR: Wear multiplier is {wear}x (1.0x = baseline at 75k mi). "
            f"At {mileage:,} miles, accumulated wear is amplifying all non-NHTSA signals significantly. "
            f"This means the risk score is elevated largely BECAUSE of high mileage, not necessarily "
            f"because the vehicle model itself is unreliable. Emphasize this in your verdict."
        )
    elif wear <= 0.6:
        lines.append(
            f"*** LOW MILEAGE BENEFIT: Wear multiplier is only {wear}x. "
            f"At {mileage:,} miles, non-NHTSA signals are heavily discounted. "
            f"The vehicle hasn't accumulated enough wear for most issues to manifest."
        )

    if floor >= 10:
        lines.append(
            f"*** MILEAGE FLOOR: The mileage-based minimum risk floor is {floor}/100. "
            f"Even a perfectly reliable model carries this baseline risk at {mileage:,} miles "
            f"due to general mechanical wear."
        )

    lines.append("\nWeighted contribution to final score (largest = most important):")
    for key, contrib in ranked:
        pct = (contrib / total * 100) if total > 0 else 0
        raw_score = sc.get(key, {}).get("score", 0) if key != "mileage_floor" else floor
        label = factor_labels.get(key, key)
        marker = " <<<< DOMINANT" if pct >= 30 else (" << significant" if pct >= 15 else "")
        lines.append(f"  - {label}: {contrib:.1f} pts ({pct:.0f}% of score){marker}")

    dominant_keys = [k for k, v in ranked if (v / total * 100) >= 20]
    minor_keys = [k for k, v in ranked if (v / total * 100) < 10]

    if dominant_keys:
        dom_labels = [factor_labels.get(k, k) for k in dominant_keys]
        lines.append(f"\nFOCUS YOUR VERDICT ON: {', '.join(dom_labels)}")
    if minor_keys:
        minor_labels = [factor_labels.get(k, k) for k in minor_keys]
        lines.append(f"DO NOT emphasize these (minor contributors): {', '.join(minor_labels)}")

    if mileage_is_dominant:
        lines.append(
            "\nIMPORTANT: Since mileage wear is a major amplifier, frame your verdict around "
            "the high mileage rather than listing individual system problems. For example: "
            "'While this model has strong reliability fundamentals, at 150,000 miles the accumulated "
            "wear means you should budget for age-related maintenance' — NOT a list of every TSB."
        )

    tsb_systems = v2_signals.get("tsb_by_system", [])
    tsb_total = v2_signals.get("tsb_total", 0)

    nhtsa_contrib_pct = (wc.get("nhtsa", 0) / total * 100) if total > 0 else 0
    tsb_contrib_pct = (wc.get("tsb", 0) / total * 100) if total > 0 else 0

    if nhtsa_contrib_pct >= 20:
        lines.append(f"\nNHTSA detail (significant contributor): {sc.get('nhtsa', {}).get('score', 0)}/100 raw score")
    if tsb_contrib_pct >= 15 and tsb_systems:
        critical_tsbs = [t for t in tsb_systems if t["category"] == "critical"][:3]
        if critical_tsbs:
            tsb_desc = ", ".join(f"{t['system']} ({t['count']})" for t in critical_tsbs)
            lines.append(f"TSB detail (significant contributor): {tsb_total} total, critical systems: {tsb_desc}")

    investigations = v2_signals.get("investigations", [])
    inv_contrib_pct = (wc.get("investigation", 0) / total * 100) if total > 0 else 0
    if inv_contrib_pct >= 10 and investigations:
        inv_desc = ", ".join(f"{inv['type_label']}" for inv in investigations[:3])
        lines.append(f"Investigation detail (significant contributor): {inv_desc}")

    return "\n".join(lines) + "\n"


def _get_tier_config(risk_score: float) -> tuple[str, str, str, str]:
    """Return (tier_label, tone_rules, summary_instruction, reasoning_instruction) for the risk score."""

    if risk_score <= 20:
        tier_label = "Excellent"
        tone_rules = """TONE RULES (EXCELLENT — score 1-20 — strictly follow):
- This vehicle is an EXCELLENT buy. Your verdict MUST be enthusiastically positive.
- ABSOLUTELY DO NOT mention ANY specific problems, failure modes, system issues, complaint counts, TSBs, or investigations.
- NEVER reference brake issues, engine problems, electrical concerns, or ANY specific component failures — not even as "things to watch."
- The buyer should walk away feeling fully CONFIDENT about this purchase.
- Focus ONLY on positives: excellent reliability profile, strong brand quality, favorable mileage, well-suited for purchase.
- The ONLY caveat you may include is a standard recommendation for a pre-purchase inspection (which applies to ANY used car)."""
        summary_instruction = (
            '"executive_summary" — 3-4 sentences. Start by enthusiastically stating this is an excellent buy. '
            'Highlight ONLY positives: outstanding reliability, strong brand reputation, favorable mileage. '
            'NEVER mention any specific problems, complaints, systems, TSBs, or failure modes. '
            'End with an encouraging recommendation to buy with high confidence. '
            'Talk directly to the buyer. No numeric scores or letter grades.'
        )
        reasoning_instruction = (
            '"verdict_reasoning" — Array of 3-4 short bullet strings explaining WHY this is an excellent buy. '
            'ONLY mention positives: very low risk, mileage advantage, brand reliability, minimal service bulletins. '
            'NEVER mention specific problems, failure types, or complaint counts for any system. '
            'Include one bullet about how the mileage works in the buyer\'s favor.'
        )

    elif risk_score <= 30:
        tier_label = "Great"
        tone_rules = """TONE RULES (GREAT — score 20-30 — strictly follow):
- This vehicle is a GREAT buy. Your verdict should be positive and reassuring.
- Do NOT mention specific problems, failure modes, or complaint counts.
- You may briefly mention the brand's general reliability standing if it's positive.
- The buyer should feel confident about this purchase.
- The only caveat is a standard pre-purchase inspection recommendation."""
        summary_instruction = (
            '"executive_summary" — 3-4 sentences. Start by stating this is a great buy. '
            'Emphasize positives: solid reliability, good brand standing, reasonable mileage. '
            'Do NOT mention specific problems or complaint counts. '
            'End with a confident recommendation to proceed. '
            'Talk directly to the buyer. No numeric scores or letter grades.'
        )
        reasoning_instruction = (
            '"verdict_reasoning" — Array of 3-4 short bullet strings explaining WHY this is a great buy. '
            'Focus on positives: low risk profile, brand reliability, mileage condition. '
            'Do NOT list any specific system failures or complaint data. '
            'Include one bullet about mileage impact.'
        )

    elif risk_score <= 45:
        tier_label = "Good — Be Cautious"
        tone_rules = """TONE RULES (GOOD BUT CAUTIOUS — score 30-45 — strictly follow):
- This vehicle is a GOOD buy, but the buyer should exercise caution.
- ONLY mention the DOMINANT factors from the score breakdown below. Do NOT list every constraint.
- If mileage wear is the dominant factor, frame the caution around high mileage and age-related wear — do NOT enumerate every TSB or system.
- If a specific system (e.g., engine, transmission) is the dominant factor, mention ONLY that system.
- Keep the tone balanced and informative — the buyer should feel this is still a solid choice.
- Do NOT be alarmist. Do NOT list complaint counts for every system.
- End with a positive recommendation to proceed with a pre-purchase inspection."""
        summary_instruction = (
            '"executive_summary" — 3-4 sentences. Start by saying this is a good buy overall. '
            'Mention ONLY the 1-2 dominant factors that are actually driving the risk score (see DOMINANT SCORE FACTORS section). '
            'If mileage is the dominant factor, say something like "at this mileage, budgeting for routine wear items is wise" — '
            'do NOT list specific TSBs or complaint systems. '
            'End with a recommendation to get a pre-purchase inspection. '
            'Talk directly to the buyer. No numeric scores or letter grades.'
        )
        reasoning_instruction = (
            '"verdict_reasoning" — Array of 3-5 short bullet strings. '
            'Start with positives (overall reliability, brand quality). '
            'Then mention ONLY the dominant factors from the score breakdown — NOT every constraint. '
            'If mileage wear is the biggest factor, dedicate a bullet to that and be lighter on system-specific issues. '
            'End with a reassuring note about the vehicle still being a good choice.'
        )

    elif risk_score <= 60:
        tier_label = "Fair — More Caution Needed"
        tone_rules = """TONE RULES (FAIR — score 45-60 — strictly follow):
- This vehicle is a FAIR buy, but the buyer needs to exercise extra caution.
- Focus your verdict on the DOMINANT factors from the score breakdown — NOT every signal.
- If mileage wear is the primary driver (e.g., a reliable brand at very high mileage), frame it as: "This is fundamentally a reliable model, but at this mileage you should budget for age-related maintenance and wear."
- If specific system problems (NHTSA complaints, TSBs, investigations) are the primary drivers, mention THOSE specific systems.
- Do NOT list every TSB system, every complaint count, and every investigation if they are minor contributors.
- Recommend a thorough pre-purchase inspection AND budgeting for potential repairs."""
        summary_instruction = (
            '"executive_summary" — 3-4 sentences. Start by saying this is a fair buy that requires extra caution. '
            'Mention ONLY the dominant factors driving the score (see DOMINANT SCORE FACTORS section). '
            'If mileage wear is the biggest driver, acknowledge the brand reliability but emphasize the high-mileage wear risk. '
            'If system-specific issues are dominant, mention those 1-2 systems specifically. '
            'End with a recommendation to get a thorough inspection and budget for potential repairs. '
            'Talk directly to the buyer. No numeric scores or letter grades.'
        )
        reasoning_instruction = (
            '"verdict_reasoning" — Array of 3-5 short bullet strings explaining WHY extra caution is needed. '
            'Focus ONLY on the dominant contributors to the score. '
            'If mileage is the primary driver, lead with that and keep system-specific bullets brief. '
            'If a specific system is the primary driver, give it a detailed bullet. '
            'Do NOT add a bullet for every signal — only the ones marked as dominant or significant.'
        )

    elif risk_score <= 75:
        tier_label = "Bad"
        tone_rules = """TONE RULES (BAD — score 60-75 — strictly follow):
- This vehicle poses SIGNIFICANT risk. Be direct and serious.
- Focus on the DOMINANT factors driving the score (see breakdown below).
- If mileage wear is a major amplifier, acknowledge it but also call out the specific system issues that are being amplified.
- Reference SPECIFIC failure modes from the dominant contributors — not every minor signal.
- The buyer needs to understand exactly what the main risks are.
- Recommend the buyer strongly consider alternatives OR budget heavily for repairs."""
        summary_instruction = (
            '"executive_summary" — 3-4 sentences. Start by clearly stating this vehicle poses significant risk. '
            'Focus on the 2-3 DOMINANT factors driving the score. '
            'If mileage is a major amplifier, mention both the mileage concern and the top 1-2 system issues. '
            'End with a clear recommendation to either look elsewhere or budget heavily for repairs. '
            'Talk directly to the buyer. No numeric scores or letter grades.'
        )
        reasoning_instruction = (
            '"verdict_reasoning" — Array of 4-5 short bullet strings explaining WHY this is a bad buy. '
            'Focus on the dominant score contributors. Give detailed bullets only for significant factors. '
            'Include one bullet about mileage impact if the wear factor is high. '
            'Include one bullet about expected repair costs for the dominant problem areas.'
        )

    else:
        tier_label = "Critical Risk"
        tone_rules = """TONE RULES (CRITICAL RISK — score >75 — strictly follow):
- This vehicle is at CRITICAL risk level. Give the strongest possible warning.
- Be completely direct about the MAJOR problems driving the score.
- Focus on the dominant factors — even at critical risk, prioritize the biggest contributors rather than listing everything.
- Explain the real-world consequences of the dominant failures (safety hazards, expensive repairs, stranding risk).
- STRONGLY recommend the buyer look at alternative vehicles instead."""
        summary_instruction = (
            '"executive_summary" — 3-4 sentences. Start with a strong warning that this vehicle carries critical risk. '
            'Reference the top 2-3 dominant factors driving the score with specifics. '
            'Explain real-world consequences (safety, reliability, cost). '
            'End by strongly recommending the buyer consider alternative vehicles. '
            'Talk directly to the buyer. No numeric scores or letter grades.'
        )
        reasoning_instruction = (
            '"verdict_reasoning" — Array of 4-5 short bullet strings with detail on the dominant risk factors. '
            'Focus on the biggest contributors — not every minor signal. '
            'Include one bullet about safety implications if relevant. '
            'Include one bullet about financial risk (expected repair costs). '
            'Include one bullet about mileage if the wear factor is high.'
        )

    return tier_label, tone_rules, summary_instruction, reasoning_instruction


def enhance_report_sections(
    vehicle: dict,
    report: dict,
    vector_complaints: list[dict] | None = None,
    bulk_stats: dict | None = None,
    price_data: dict | None = None,
    v2_signals: dict | None = None,
) -> dict:
    """Generate the Buyer's Verdict (executive_summary + verdict_reasoning).

    Uses V2 signal data (TSBs, investigations, MFR comms, brand reliability) alongside
    NHTSA complaint data to give the LLM full context for its assessment.
    """
    sections = report.get("sections", {})
    vs = sections.get("vehicle_summary", {})
    cr = sections.get("current_risk", {})
    oe = sections.get("owner_experience", {})

    top_issues_detail = [
        {
            "system": i.get("system"),
            "description": i.get("description", "")[:250],
            "probability": i.get("probability"),
            "complaint_count": i.get("complaint_count", 0),
            "severity": i.get("severity", 0),
            "phase": i.get("phase", "unknown"),
        }
        for i in cr.get("top_issues", [])[:5]
    ]

    severe_issues = [
        i for i in cr.get("top_issues", [])
        if i.get("severity", 0) >= 8
    ][:5]

    owner_reports = [
        r[:200] for r in oe.get("sample_reports", [])[:8]
    ]

    recalls_brief = [
        {"component": r.get("component"), "summary": r.get("summary", "")[:100]}
        for r in vs.get("recalls", [])[:3]
    ]

    mileage = vehicle.get("mileage", 0)
    phase_summary = cr.get("phase_summary", {})
    risk_score = vs.get("reliability_risk_score", 0)
    mileage_assessment = _mileage_assessment(mileage, risk_score, phase_summary)

    tier_label, tone_rules, summary_instruction, reasoning_instruction = _get_tier_config(risk_score)

    context = {
        "vehicle": _vehicle_str(vehicle),
        "mileage": mileage,
        "mileage_assessment": mileage_assessment,
        "risk_tier": tier_label,
        "internal_risk_score": risk_score,
        "phase_distribution": phase_summary,
        "total_complaints": vs.get("total_complaints"),
        "total_recalls": vs.get("total_recalls"),
        "open_recalls": recalls_brief,
    }

    if risk_score > 45:
        context["top_issues"] = top_issues_detail
        context["severe_issues"] = severe_issues
        context["owner_reports"] = owner_reports
    elif risk_score > 30:
        context["top_issues"] = [
            {"system": i["system"], "complaint_count": i["complaint_count"]}
            for i in top_issues_detail[:3]
        ]

    dominant_factors_section = _analyze_dominant_factors(v2_signals, risk_score, mileage)

    rag_section = ""
    if bulk_stats:
        tc = bulk_stats.get("total_complaints", 0)
        pct = bulk_stats.get("complaints_percentile", 50)
        avg = bulk_stats.get("global_mean_complaints", 0)
        interp = bulk_stats.get("interpretation", "")
        rag_section = (
            f"\nComplaint context: {tc} NHTSA complaints ({pct:.0f}th percentile, "
            f"average is {avg:.0f}). {interp}.\n"
        )

    pricing_section = ""
    pricing = report.get("sections", {}).get("pricing", {})
    if pricing.get("available"):
        avg_market = pricing.get("avg_market_price", 0)
        asking = pricing.get("asking_price")
        source = pricing.get("source", "")
        count = pricing.get("listings_count", 0)
        match_lvl = pricing.get("match_level", "estimate")

        pricing_section = f"\nMarket price data: Average market price is ${avg_market:,} (source: {source}, {count} listings, match: {match_lvl})."
        if asking:
            diff = asking - avg_market
            if diff > 0:
                pricing_section += f" Asking price ${asking:,} is ${diff:,} ABOVE market average."
            elif diff < 0:
                pricing_section += f" Asking price ${asking:,} is ${abs(diff):,} BELOW market average."
            else:
                pricing_section += f" Asking price ${asking:,} matches market average."

        dom = pricing.get("days_on_market")
        if dom:
            if dom > 60:
                pricing_section += f" These vehicles sit on the market an average of {dom} days — slow sellers, which gives buyers negotiating leverage."
            elif dom > 30:
                pricing_section += f" Average days on market is {dom} — moderate demand."
            else:
                pricing_section += f" Average days on market is only {dom} — high demand, these sell fast."

        pp = pricing.get("price_position")
        if pp:
            pricing_section += f" The asking price falls at the {pp['percentile']:.0f}th percentile ({pp['label']}) of comparable listings."

        pricing_section += "\n"

    price_instruction = ""
    if pricing.get("available") and pricing.get("avg_market_price"):
        if pricing.get("asking_price"):
            dom_hint = ""
            if pricing.get("days_on_market"):
                dom = pricing["days_on_market"]
                if dom > 60:
                    dom_hint = f" Mention that these vehicles average {dom} days on market (slow sellers = negotiating leverage)."
                elif dom <= 30:
                    dom_hint = f" Note these sell quickly (avg {dom} days on market) so the buyer may need to act fast."
            pos_hint = ""
            pp = pricing.get("price_position")
            if pp:
                pos_hint = f' The asking price is at the {pp["percentile"]:.0f}th percentile ({pp["label"]}) of comparable listings — mention this.'
            price_instruction = (
                '\nPRICING: Market price data is available. Include a brief comment on '
                'the asking price vs market average in your executive_summary (e.g. "priced '
                '$X below/above market average"). Add one bullet in verdict_reasoning about the price value.'
                + dom_hint + pos_hint
            )
        else:
            price_instruction = (
                '\nPRICING: Market price data is available but no asking price was provided. '
                'You may briefly mention the average market price as context if relevant.'
            )

    prompt = f"""You are an expert automotive advisor helping a car buyer evaluate a used {_vehicle_str(vehicle)}.

RISK TIER: {tier_label} (score: {risk_score}/100 — higher = more risky)
Risk score ranges: 1-20 Excellent, 20-30 Great, 30-45 Good (be cautious), 45-60 Fair (more caution), 60-75 Bad, >75 Critical Risk.

MILEAGE RULES (strictly follow):
- Mileage tier: "{mileage_assessment['tier_label']}" ({mileage:,} miles)
- {mileage_assessment['guidance']}
- Lower mileage = more favorable verdict. Never call a low-mileage car "risky" when the same car at higher mileage would be rated better.
- "Upcoming" issues on a low-mileage car are FUTURE concerns, not current problems. This is positive.
- "Past" issues on a high-mileage car represent accumulated wear.

{tone_rules}

CRITICAL RULE — FOCUS ON DOMINANT FACTORS ONLY:
Your verdict MUST focus on the factors that are actually driving the risk score, NOT list every constraint.
The "DOMINANT SCORE FACTORS" section below tells you exactly which factors matter most.
If mileage wear is the dominant factor, frame your verdict around mileage — do NOT list every TSB or complaint.
If NHTSA complaints are the dominant factor, focus on the specific complaint patterns.
Only mention a factor if it is marked as "DOMINANT" or "significant" below.
Minor contributors should be ignored in your verdict text.
{dominant_factors_section}
Phase distribution: {json.dumps(phase_summary)}
(past = already through that failure zone, current = in the zone, upcoming = approaching, future = far away)
{rag_section}{pricing_section}{price_instruction}
Vehicle data:
{json.dumps(context, indent=2)}

Return ONLY a valid JSON object with exactly these two keys:

1. {summary_instruction}

2. {reasoning_instruction}"""

    text = _llm_call(prompt, max_tokens=1024)

    trace = get_trace()

    if text is None:
        if trace:
            trace.log_llm_call(
                purpose="buyers_verdict",
                prompt=prompt,
                response_raw=None,
                response_parsed=None,
                status="llm_returned_none",
            )
        return report

    result = _extract_json(text)
    if not isinstance(result, dict):
        logger.warning("Could not parse buyer's verdict LLM response")
        if trace:
            trace.log_llm_call(
                purpose="buyers_verdict",
                prompt=prompt,
                response_raw=text,
                response_parsed=None,
                status="parse_failed",
            )
        return report

    logger.info("Buyer's verdict generated successfully")
    if trace:
        trace.log_llm_call(
            purpose="buyers_verdict",
            prompt=prompt,
            response_raw=text,
            response_parsed=result,
            status="success",
        )

    if result.get("executive_summary"):
        sections["executive_summary"] = {
            "text": result["executive_summary"],
            "verdict_reasoning": result.get("verdict_reasoning", []),
            "llm_enhanced": True,
        }

    return report
