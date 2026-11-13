"""Human Annotation Rubric for Financial News Sentiment Classification.

Adheres strictly to PRD Section 12.1.
"""

RUBRIC_GUIDELINES = """
# Financial Headline Sentiment Annotation Rubric (3-Class)

## Principles
1. Perspective: Label strictly from the perspective of an equity investor or financial analyst reading the headline alone.
2. No Extraneous Forecasts: Base the label solely on the explicit information contained in the headline; do not extrapolate speculative macro consequences or unmentioned stock movements.
3. Target Ambiguity: If multiple corporate entities are mentioned with opposite fortunes (e.g. 'Company A sues Company B'), assign the label for the primary subject, or mark ambiguous.

## Class Definitions

### 1. Positive
Signals a favorable corporate, operational, or market development:
- Earnings, revenues, or margins materially above analyst consensus or historical levels.
- Raised forward earnings guidance or upward revisions in fiscal outlook.
- Major regulatory clearances, drug approvals (FDA/EMA), or favorable antitrust decisions.
- Lucrative multi-year customer contracts, significant expansion of enterprise backlog.
- Capital return enhancements: dividend hikes, major share repurchase authorizations.
- Successful debt restructuring, credit rating upgrades.

### 2. Negative
Signals an adverse corporate, operational, or market development:
- Earnings, revenues, or cash flows missing consensus expectations.
- Downward revision of guidance, profit warnings, margin compression warnings.
- Regulatory penalties, antitrust lawsuits, fraud investigations, SEC inquiries.
- Material customer losses, contract cancellations, or supply chain bottlenecks.
- Executive resignations under duress, accounting irregularities, restatements.
- Workforce reductions driven by distress, credit rating downgrades, debt covenant breaches.
- Product recalls, catastrophic facility failures, patent expirations without pipeline replacement.

### 3. Neutral
Objective, factual, or balanced information without unambiguous directional sentiment:
- Routine corporate schedules: earnings call announcements, AGM dates.
- Neutral macroeconomic indicators: Fed policy holds meeting, CPI matches exact expectation.
- Balanced developments: 'Revenue rises 5% while net income declines 4%'.
- Mergers and acquisitions at market price prior to regulatory or shareholder reaction.
- Industry commentary, executive appointments during ordinary succession.
"""

