"""Email service — renders the report page to PDF via headless browser and sends via Resend."""

from __future__ import annotations

import logging
from typing import Optional

import resend

from config.settings import RESEND_API_KEY, RESEND_FROM_EMAIL

logger = logging.getLogger(__name__)

resend.api_key = RESEND_API_KEY


def _risk_color(score: float) -> str:
    if score >= 70:
        return "#ef4444"
    if score >= 40:
        return "#f59e0b"
    return "#22c55e"


def _risk_label(score: float) -> str:
    if score >= 70:
        return "High Risk"
    if score >= 40:
        return "Moderate Risk"
    return "Low Risk"


def generate_report_pdf(report_url: str) -> bytes | None:
    """Use Playwright to render the report page and print it to PDF,
    identical to the browser's Print → Save as PDF."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(report_url, wait_until="networkidle")
            page.wait_for_timeout(2000)

            page.evaluate("""() => {
                document.querySelectorAll('[x-data]').forEach(el => {
                    const stack = el._x_dataStack || (el.__x ? [el.__x.$data] : []);
                    stack.forEach(scope => {
                        if ('open' in scope) scope.open = true;
                        if ('ownerOpen' in scope) scope.ownerOpen = true;
                        if ('showAll' in scope) scope.showAll = true;
                        if ('showAllRecalls' in scope) scope.showAllRecalls = true;
                    });
                });
            }""")
            page.wait_for_timeout(500)

            pdf_bytes = page.pdf(
                format="Letter",
                print_background=True,
                margin={"top": "0.5in", "bottom": "0.5in", "left": "0.5in", "right": "0.5in"},
            )
            browser.close()
            logger.info("PDF generated via Playwright (%d bytes)", len(pdf_bytes))
            return pdf_bytes
    except Exception:
        logger.exception("Playwright PDF generation failed")
        return None


def _build_marketing_email(report: dict, report_url: str) -> str:
    vehicle = report.get("vehicle", {})
    vs = report.get("sections", {}).get("vehicle_summary", {})
    title = f"{vehicle.get('year', '')} {vehicle.get('make', '')} {vehicle.get('model', '')}"
    risk_score = vs.get("reliability_risk_score", 0)
    complaints = vs.get("total_complaints", 0)
    recalls = vs.get("total_recalls", 0)
    color = _risk_color(risk_score)
    label = _risk_label(risk_score)

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#f9fafb;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
<div style="max-width:600px;margin:0 auto;padding:20px;">

    <!-- Header -->
    <div style="background:linear-gradient(135deg,#111111,#1a1a2e);border-radius:16px 16px 0 0;padding:30px 30px 24px;text-align:center;">
        <h1 style="color:#FFC700;font-size:24px;font-weight:800;margin:0;">CarAdvisr</h1>
        <p style="color:#6b7280;font-size:12px;margin:6px 0 0;">RISK INTELLIGENCE</p>
    </div>

    <!-- Body -->
    <div style="background:#ffffff;padding:32px;border-radius:0 0 16px 16px;box-shadow:0 4px 24px rgba(0,0,0,0.06);">

        <h2 style="font-size:20px;font-weight:800;color:#111;margin:0;">Your report for the {title} is ready.</h2>
        <p style="font-size:14px;color:#6b7280;margin:8px 0 24px;line-height:1.6;">We analyzed <strong style="color:#111;">{complaints:,} owner complaints</strong>, <strong style="color:#111;">{recalls} safety recalls</strong>, service bulletins, and federal investigation records to give you a clear picture of this vehicle's reliability.</p>

        <!-- Score highlight -->
        <div style="background:#fafafa;border:1px solid #e5e7eb;border-radius:12px;padding:20px;text-align:center;margin-bottom:24px;">
            <table width="100%" cellpadding="0" cellspacing="0">
                <tr>
                    <td style="text-align:center;width:33%;vertical-align:top;">
                        <div style="font-size:36px;font-weight:900;color:{color};">{risk_score}</div>
                        <div style="font-size:11px;color:#6b7280;margin-top:2px;">Risk Score</div>
                        <div style="display:inline-block;margin-top:6px;background:{color}18;color:{color};font-size:11px;font-weight:700;padding:3px 10px;border-radius:20px;">{label}</div>
                    </td>
                    <td style="text-align:center;width:33%;vertical-align:top;">
                        <div style="font-size:36px;font-weight:900;color:#111;">{complaints:,}</div>
                        <div style="font-size:11px;color:#6b7280;margin-top:2px;">Complaints</div>
                    </td>
                    <td style="text-align:center;width:33%;vertical-align:top;">
                        <div style="font-size:36px;font-weight:900;color:#111;">{recalls}</div>
                        <div style="font-size:11px;color:#6b7280;margin-top:2px;">Recalls</div>
                    </td>
                </tr>
            </table>
        </div>

        <p style="font-size:14px;color:#374151;line-height:1.6;margin:0 0 6px;">Your complete report is attached as a PDF. It includes:</p>
        <table cellpadding="0" cellspacing="0" style="margin-bottom:24px;">
            <tr><td style="padding:4px 0;font-size:14px;color:#374151;">&#10003;&nbsp;&nbsp;AI-powered buyer's verdict</td></tr>
            <tr><td style="padding:4px 0;font-size:14px;color:#374151;">&#10003;&nbsp;&nbsp;Pre-purchase inspection checklist</td></tr>
            <tr><td style="padding:4px 0;font-size:14px;color:#374151;">&#10003;&nbsp;&nbsp;Risk signal breakdown by data source</td></tr>
            <tr><td style="padding:4px 0;font-size:14px;color:#374151;">&#10003;&nbsp;&nbsp;Safety recall details &amp; remedies</td></tr>
            <tr><td style="padding:4px 0;font-size:14px;color:#374151;">&#10003;&nbsp;&nbsp;Real owner complaint summaries</td></tr>
        </table>

        <!-- CTA -->
        <div style="text-align:center;margin-bottom:24px;">
            <a href="{report_url}" style="display:inline-block;background:#FFC700;color:#111111;font-size:15px;font-weight:700;padding:14px 36px;border-radius:12px;text-decoration:none;">View Interactive Report &rarr;</a>
        </div>

        <p style="font-size:12px;color:#9ca3af;text-align:center;margin:0;">The interactive version includes live charts and additional data visualizations.</p>
    </div>

    <!-- Footer -->
    <div style="text-align:center;padding:20px 0;">
        <p style="font-size:11px;color:#9ca3af;margin:0 0 4px;">Making smarter car-buying decisions with data.</p>
        <p style="font-size:10px;color:#c4c4c4;margin:0;"><a href="{report_url}" style="color:#c4c4c4;">caradvisr.com</a></p>
    </div>
</div>
</body>
</html>"""


def send_report_email(
    to_email: str,
    report: dict,
    report_id: str,
    base_url: str = "",
) -> Optional[dict]:
    """Render the report page to PDF via headless browser and email it."""
    if not RESEND_API_KEY:
        logger.warning("RESEND_API_KEY not set, skipping email send")
        return None

    vehicle = report.get("vehicle", {})
    title = f"{vehicle.get('year', '')} {vehicle.get('make', '')} {vehicle.get('model', '')}"
    report_url = f"{base_url}/report/{report_id}" if base_url else f"/report/{report_id}"

    html = _build_marketing_email(report, report_url)

    pdf_bytes = generate_report_pdf(report_url)
    if not pdf_bytes:
        logger.warning("PDF generation failed, sending email without attachment")

    vs = report.get("sections", {}).get("vehicle_summary", {})
    complaints = vs.get("total_complaints", 0)
    safe_title = title.replace(" ", "_")
    filename = f"{safe_title}_Risk_Report_—_{complaints}_Complaints.pdf"

    try:
        params: resend.Emails.SendParams = {
            "from": f"CarAdvisr <{RESEND_FROM_EMAIL}>",
            "to": [to_email],
            "subject": f"Your {title} Risk Report Is Ready",
            "html": html,
        }
        if pdf_bytes:
            params["attachments"] = [
                {
                    "filename": filename,
                    "content": list(pdf_bytes),
                }
            ]
        response = resend.Emails.send(params)
        logger.info("Report email sent to %s (id=%s, pdf=%s)", to_email, response.get("id"), bool(pdf_bytes))
        return response
    except Exception:
        logger.exception("Failed to send report email to %s", to_email)
        return None
