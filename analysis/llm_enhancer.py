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


def enhance_report_sections(
    vehicle: dict,
    report: dict,
    vector_complaints: list[dict] | None = None,
    bulk_stats: dict | None = None,
    price_data: dict | None = None,
) -> dict:
    """Generate only the Buyer's Verdict (executive_summary + verdict_reasoning).

    All other sections are populated from raw data by the report generator
    without LLM calls, keeping report generation fast.
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

    context = {
        "vehicle": _vehicle_str(vehicle),
        "mileage": mileage,
        "mileage_assessment": mileage_assessment,
        "internal_risk_score": risk_score,
        "phase_distribution": phase_summary,
        "total_complaints": vs.get("total_complaints"),
        "total_recalls": vs.get("total_recalls"),
        "open_recalls": recalls_brief,
    }

    if risk_score > 40:
        context["top_issues"] = top_issues_detail
        context["severe_issues"] = severe_issues
        context["owner_reports"] = owner_reports
    elif risk_score >= 20:
        context["top_issues"] = [
            {"system": i["system"], "complaint_count": i["complaint_count"]}
            for i in top_issues_detail[:3]
        ]

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
        pricing_section += "\n"

    if risk_score > 40:
        tone_rules = """TONE RULES (HIGH RISK — strictly follow):
- This vehicle is HIGH RISK. Be direct about the problems.
- Reference SPECIFIC failure modes from owner reports (e.g. "connecting rod bearing wear", "steering lock-up") rather than generic system names.
- Mention complaint counts for the worst systems.
- The buyer needs to understand exactly what they're getting into."""
        summary_instruction = (
            '"executive_summary" — 3-4 sentences. Start by clearly stating this is a risky buy. '
            'Reference SPECIFIC failure modes from owner reports with complaint counts. '
            'End with a clear recommendation (look elsewhere, or budget heavily for repairs). '
            'Talk directly to the buyer. No numeric scores or letter grades.'
        )
        reasoning_instruction = (
            '"verdict_reasoning" — Array of 3-5 short bullet strings explaining WHY. '
            'Each bullet MUST cite specific failure types from the owner reports and complaint data '
            '(not just "engine issues" — say what the actual failures are). '
            'Include complaint counts, recall status, and one bullet about mileage impact.'
        )
    elif risk_score >= 20:
        tone_rules = """TONE RULES (FAIR RISK — strictly follow):
- This vehicle is a FAIR buy — not perfect, but not a dealbreaker.
- Do NOT list out every problem. Instead, briefly mention 1-2 areas to watch (e.g., "proceed with caution on potential engine concerns").
- Keep the tone balanced and encouraging — the buyer should feel informed, not scared.
- Focus on what makes it a reasonable choice and what to keep an eye on."""
        summary_instruction = (
            '"executive_summary" — 3-4 sentences. Start by saying this is a fair/reasonable buy. '
            'Briefly mention 1-2 areas to be cautious about (e.g., "keep an eye on engine health") '
            'without listing specific failure modes or complaint counts. '
            'End with a positive recommendation to proceed with a pre-purchase inspection. '
            'Talk directly to the buyer. No numeric scores or letter grades.'
        )
        reasoning_instruction = (
            '"verdict_reasoning" — Array of 3-5 short bullet strings explaining WHY. '
            'Keep bullets balanced — mention positives (mileage, recalls addressed, etc.) alongside '
            '1-2 areas of caution. Do NOT list out all problems or complaint counts for every system. '
            'Include one bullet about mileage impact.'
        )
    else:
        tone_rules = """TONE RULES (LOW RISK — STRICTLY follow, this is CRITICAL):
- This vehicle is a GOOD buy. Your verdict MUST be positive and reassuring.
- ABSOLUTELY DO NOT mention ANY specific problems, failure modes, system issues, or complaint counts. Not even as "things to watch."
- NEVER reference brake issues, engine problems, electrical concerns, or ANY specific component failures.
- The buyer should walk away feeling CONFIDENT about this purchase.
- Focus ONLY on positives: low risk score, favorable mileage, well-suited for purchase, good value.
- The ONLY caveat you may include is a standard recommendation for a pre-purchase inspection (which applies to ANY used car)."""
        summary_instruction = (
            '"executive_summary" — 3-4 sentences. Start by confidently stating this is a good buy. '
            'Highlight ONLY positives: low risk profile, favorable mileage, solid choice. '
            'NEVER mention any specific problems, complaints, systems, or failure modes. '
            'End with an encouraging recommendation to buy with confidence. '
            'Talk directly to the buyer. No numeric scores or letter grades.'
        )
        reasoning_instruction = (
            '"verdict_reasoning" — Array of 3-4 short bullet strings explaining WHY this is a good buy. '
            'ONLY mention positives: low risk score, mileage advantage, overall reliability. '
            'NEVER mention specific problems, failure types, or complaint counts for any system. '
            'Include one bullet about how the mileage works in the buyer\'s favor.'
        )

    price_instruction = ""
    if pricing.get("available") and pricing.get("avg_market_price"):
        if pricing.get("asking_price"):
            price_instruction = (
                '\nPRICING: Market price data is available. Include a brief comment on '
                'the asking price vs market average in your executive_summary (e.g. "priced '
                '$X below/above market average"). Add one bullet in verdict_reasoning about the price value.'
            )
        else:
            price_instruction = (
                '\nPRICING: Market price data is available but no asking price was provided. '
                'You may briefly mention the average market price as context if relevant.'
            )

    prompt = f"""You are an expert automotive advisor helping a car buyer evaluate a used {_vehicle_str(vehicle)}.

MILEAGE RULES (strictly follow):
- Mileage tier: "{mileage_assessment['tier_label']}" ({mileage:,} miles)
- {mileage_assessment['guidance']}
- Risk score: {mileage_assessment['risk_score_out_of_100']}/100 (higher = more risky)
- Lower mileage = more favorable verdict. Never call a low-mileage car "risky" when the same car at higher mileage would be rated better.
- "Upcoming" issues on a low-mileage car are FUTURE concerns, not current problems. This is positive.
- "Past" issues on a high-mileage car represent accumulated wear.

{tone_rules}

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
