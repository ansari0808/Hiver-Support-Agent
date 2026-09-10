"""Central config: paths, brand, intent taxonomy, model choice.

Keeping every tunable in one place is deliberate — see DECISION_LOG.md #2.
"""
from pathlib import Path

# ---- paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_GOLDEN = ROOT / "data" / "golden"

RAW_TWCS_CSV = DATA_RAW / "twcs.csv"

# ---- brand ------------------------------------------------------------------
BRAND = "AmazonHelp"          # the Twitter handle of the brand agent, no leading @
BRAND_AUTHOR_ID = "AmazonHelp"

# ---- models -----------------------------------------------------------------
# Cheap/fast model for classification + judging, stronger model for drafting replies.
CLASSIFIER_MODEL = "claude-haiku-4-5-20251001"
GENERATION_MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = "claude-sonnet-4-6"

# ---- intent taxonomy ----------------------------------------------------------
# Derived by reading ~150 raw AmazonHelp threads and clustering by hand before
# writing this list. See DECISION_LOG.md #4 for why 8 and not 20+.
INTENT_TAXONOMY = {
    "order_status": "Where is my order / delivery ETA / tracking questions",
    "refund_or_return": "Requesting a refund, return, or reporting a wrong/damaged item",
    "account_access": "Login, password reset, account locked/suspended, 2FA issues",
    "billing_or_charge": "Unexpected charge, double charge, subscription billing issue",
    "product_defect_or_quality": "Product broke, doesn't work, quality complaint (not shipping-related)",
    "app_or_website_bug": "App crash, website error, feature not working technically",
    "general_inquiry": "Pre-purchase question, policy question, other non-issue question",
    "complaint_escalation": "Angry/frustrated customer, repeated unresolved issue, threat to cancel/leave review",
}

INTENT_LABELS = list(INTENT_TAXONOMY.keys())

# Intents that should never be auto-handled regardless of retrieval confidence.
HIGH_RISK_INTENTS = {"billing_or_charge", "account_access", "complaint_escalation"}

# ---- retrieval ----------------------------------------------------------------
TOP_K_RETRIEVAL = 3
MIN_RETRIEVAL_SIMILARITY_FOR_AUTOHANDLE = 0.18  # tuned empirically, see Decision Log #7

# ---- routing ------------------------------------------------------------------
# Keyword triggers that force escalation regardless of everything else (safety net).
FORCE_ESCALATE_KEYWORDS = [
    "lawyer", "legal action", "sue", "fraud", "unauthorized charge",
    "suicide", "self harm", "kill myself", "hurt myself",
    "bbb complaint", "chargeback", "class action",
]
