"""Streamlit interface for the Hiver support-agent pipeline."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src import config
from src.pipeline import run_one
from src.retrieval import load_retriever


# ---------------------------------------------------------------------------
# Paths and environment
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

THREADS_PATH = (
    config.DATA_PROCESSED
    / f"threads_{config.BRAND}.jsonl"
)

# Load environment variables from .env
load_dotenv(ROOT / ".env")


# ---------------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Hiver Support Agent",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --ink: #182329;
        --muted: #64747a;
        --line: #dbe3e1;
        --orange: #ff8a5b;
    }

    html,
    body,
    [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--ink);
    }

    .stApp,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp p,
    .stApp label,
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] * {
        color: var(--ink) !important;
    }

    .stApp .kicker,
    .stApp .hero p,
    .stApp .stat-label,
    .stApp .result-label,
    .stApp [data-testid="stCaptionContainer"] p {
        color: var(--muted) !important;
    }

    .stApp .kicker {
        color: var(--orange) !important;
    }

    .stApp {
        background: #f5f7f4;
    }

    [data-testid="stSidebar"] {
        background: #e8f0eb;
        border-right: 1px solid var(--line);
    }

    .hero {
        padding: 2.2rem 0 1.2rem;
        border-bottom: 1px solid var(--line);
    }

    .kicker {
        color: var(--orange);
        font: 500 0.78rem 'DM Mono', monospace;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    h1 {
        font-size: clamp(2.2rem, 5vw, 4.5rem) !important;
        line-height: 0.98 !important;
        letter-spacing: -0.04em !important;
        margin: 0.35rem 0 !important;
    }

    .hero p {
        color: var(--muted);
        font-size: 1.05rem;
        max-width: 42rem;
    }

    .stat {
        background: white;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem;
        min-height: 6.5rem;
    }

    .stat-label {
        color: var(--muted);
        font: 500 0.7rem 'DM Mono', monospace;
        text-transform: uppercase;
    }

    .stat-value {
        font-size: 1.45rem;
        font-weight: 600;
        margin-top: 0.65rem;
    }

    .result {
        background: white;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1.2rem;
        min-height: 8rem;
    }

    .result-label {
        color: var(--muted);
        font: 500 0.7rem 'DM Mono', monospace;
        text-transform: uppercase;
    }

    .result-value {
        font-size: 1.25rem;
        font-weight: 600;
        margin-top: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def stat(label: str, value: str) -> None:
    """Render a metric/status card."""
    st.markdown(
        f"""
        <div class="stat">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_status() -> tuple[bool, bool, bool]:
    """
    Check whether the required application dependencies are available.

    Returns:
        index_ready: Processed thread index exists.
        csv_ready: Source CSV exists.
        key_ready: Anthropic API key is configured.
    """

    # Check processed thread index.
    index_ready = THREADS_PATH.is_file()

    # Check source CSV.
    csv_path = getattr(config, "RAW_TWCS_CSV", None)

    if csv_path is not None:
        csv_ready = Path(csv_path).is_file()
    else:
        csv_ready = False

    # Check Anthropic API key.
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

    # We only check that a non-empty key exists here.
    # The actual API call will validate whether the key is valid.
    key_ready = bool(api_key)

    return index_ready, csv_ready, key_ready


# ---------------------------------------------------------------------------
# Application status
# ---------------------------------------------------------------------------

index_ready, csv_ready, key_ready = get_status()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown("### HIVER / SUPPORT OPS")
st.sidebar.caption("Hiver response intelligence")
st.sidebar.divider()

st.sidebar.markdown("**Pipeline stages**")

for stage in (
    "Intent classification",
    "Thread retrieval",
    "Reply drafting",
    "Human routing",
):
    st.sidebar.markdown(f"`01`  {stage}")

st.sidebar.divider()

st.sidebar.caption(
    "Model-powered analysis uses the Anthropic API. "
    "Your key stays in the local .env file."
)


# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------

st.html(
    """
    <div class="hero">
        <div class="kicker">
            Customer support intelligence / v1
        </div>

        <h1>
            Turn a support message<br>
            into the next best action.
        </h1>

        <p>
            Classify the issue, find the closest resolved thread,
            draft a grounded reply, and decide when a human should step in.
        </p>
    </div>
    """,
)


# ---------------------------------------------------------------------------
# Status metrics
# ---------------------------------------------------------------------------

metrics = st.columns(3)

with metrics[0]:
    stat(
        "Thread index",
        "Ready" if index_ready else "Missing",
    )

with metrics[1]:
    stat(
        "Source CSV",
        "Ready" if csv_ready else "Not added",
    )

with metrics[2]:
    stat(
        "AI connection",
        "Connected" if key_ready else "Needs key",
    )


# ---------------------------------------------------------------------------
# Live analysis
# ---------------------------------------------------------------------------

st.markdown("### Live analysis")

message = st.text_area(
    "Customer message",
    value=(
        "Where is my order? It was supposed to arrive yesterday "
        "and tracking has not updated."
    ),
    height=130,
    placeholder="Paste a customer support message...",
)

analyze = st.button(
    "Analyze message",
    type="primary",
)


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

if analyze:

    # Check thread index.
    if not index_ready:
        st.error(
            "The thread index is missing. Run:\n\n"
            "python -m src.data_prep "
            "--brand AmazonHelp "
            "--max_threads 4000"
        )

    # Check API key.
    elif not key_ready:
        st.warning(
            "The interface is ready, but live analysis needs "
            "ANTHROPIC_API_KEY in .env."
        )

    # Check message.
    elif not message.strip():
        st.warning("Enter a customer message first.")

    else:
        try:
            with st.spinner(
                "Classifying, retrieving, drafting, and routing..."
            ):
                retriever = load_retriever(config.BRAND)

                result = run_one(
                    message.strip(),
                    retriever,
                )

            # Save result so it remains visible after Streamlit reruns.
            st.session_state["result"] = result

        except RuntimeError as exc:
            st.error(
                "Analysis failed. Check that .env contains "
                "a valid Anthropic API key."
            )
            st.caption(str(exc))

        except Exception as exc:
            st.error(
                "An unexpected error occurred while running the pipeline."
            )
            st.exception(exc)


# ---------------------------------------------------------------------------
# Display result
# ---------------------------------------------------------------------------

result = st.session_state.get("result")


if result:

    st.markdown("### Decision brief")

    cards = st.columns(3)

    # -----------------------------------------------------------------------
    # Intent
    # -----------------------------------------------------------------------

    with cards[0]:

        intent = result.get(
            "intent",
            "Unknown",
        )

        intent_rationale = result.get(
            "intent_rationale",
            "",
        )

        st.markdown(
            f"""
            <div class="result">
                <div class="result-label">Intent</div>
                <div class="result-value">{intent}</div>
                <p>{intent_rationale}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # Routing
    # -----------------------------------------------------------------------

    with cards[1]:

        routing_decision = result.get(
            "routing_decision",
            "Unknown",
        )

        routing_reason = result.get(
            "routing_reason",
            "",
        )

        st.markdown(
            f"""
            <div class="result">
                <div class="result-label">Routing</div>
                <div class="result-value">{routing_decision}</div>
                <p>{routing_reason}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # Grounding
    # -----------------------------------------------------------------------

    with cards[2]:

        retrieved_examples = result.get(
            "retrieved_examples",
            [],
        )

        if retrieved_examples:
            try:
                similarity = float(
                    retrieved_examples[0].get(
                        "similarity",
                        0.0,
                    )
                )
            except (TypeError, ValueError):
                similarity = 0.0
        else:
            similarity = 0.0

        is_grounded = bool(
            result.get(
                "is_grounded",
                False,
            )
        )

        grounding_status = (
            "Grounded"
            if is_grounded
            else "Uncertain"
        )

        st.markdown(
            f"""
            <div class="result">
                <div class="result-label">Grounding</div>
                <div class="result-value">
                    {grounding_status}
                </div>
                <p>
                    Top retrieval similarity:
                    {similarity:.3f}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # Draft reply + retrieved resolutions
    # -----------------------------------------------------------------------

    left, right = st.columns(
        [1.05, 0.95]
    )

    # Draft reply
    with left:

        st.markdown("#### Draft reply")

        draft_reply = result.get(
            "draft_reply",
            "No draft reply was generated.",
        )

        st.info(draft_reply)

    # Retrieved resolutions
    with right:

        st.markdown("#### Retrieved resolutions")

        if retrieved_examples:

            for i, example in enumerate(
                retrieved_examples,
                start=1,
            ):

                try:
                    example_similarity = float(
                        example.get(
                            "similarity",
                            0.0,
                        )
                    )
                except (TypeError, ValueError):
                    example_similarity = 0.0

                customer_text = example.get(
                    "customer_text",
                    "",
                )

                brand_reply_text = example.get(
                    "brand_reply_text",
                    "",
                )

                with st.expander(
                    f"Resolution {i} / "
                    f"similarity {example_similarity:.3f}"
                ):

                    st.write(customer_text)

                    if brand_reply_text:
                        st.caption(
                            brand_reply_text
                        )

        else:

            st.caption(
                "No similar resolved threads were retrieved."
            )


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

else:

    st.markdown("### What you will see here")

    st.caption(
        "Run the local data-preparation command to build "
        "the thread index, then add your API key to unlock "
        "live analysis."
    )