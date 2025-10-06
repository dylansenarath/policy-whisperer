# streamlit_app.py
import streamlit as st
from rag_core import health_check, has_index, ask_with_sources, build_or_refresh_index

st.set_page_config(page_title="Policy Whisperer (RAG) – MVP", page_icon="🩺", layout="wide")

st.title("🩺 Policy Whisperer (RAG) – MVP")
st.warning(
    "This assistant summarizes internal policy documents to support clinical decision-making. "
    "It is **not** medical advice. Verify against official policies/protocols.",
    icon="⚠️",
)

with st.sidebar:
    st.header("Ingest (Company-controlled)")
    st.write("Click to (re)build the index from the fixed internal sources (URLs + PDFs).")
    build_clicked = st.button("Build / Refresh Index")
    if build_clicked:
        with st.spinner("Indexing corpus…"):
            n, msg = build_or_refresh_index()
        if n > 0:
            st.success(msg)
        else:
            st.error(msg)

    st.divider()
    st.header("Health Check")
    if st.button("Run health check"):
        hc = health_check()
        if hc["ok"]:
            st.success("Health check passed.")
        else:
            st.error("Health check found issues. See details below.")
        for c in hc["checks"]:
            (st.write if c["ok"] else st.warning)(c["msg"])

st.subheader("Ask a policy question")
disabled_reason = None
if not has_index():
    disabled_reason = "No index yet. Click 'Build / Refresh Index' in the sidebar."

q = st.text_input(
    "Your question:",
    placeholder="e.g., What are the steps for pre-procedure hand hygiene?",
    disabled=disabled_reason is not None,
    help=disabled_reason or "Ready."
)

if disabled_reason:
    st.info(disabled_reason)
elif q:
    with st.spinner("Thinking…"):
        result = ask_with_sources(q)

    st.markdown("### Answer")
    st.write(result["answer"])

    # Render sources with aligned tags [S#]
    if result.get("sources"):
        st.markdown("**Sources**")
        for s in result["sources"]:
            tags = ", ".join(s.get("tags", []))
            title = s.get("title") or "source"
            url = s.get("url") or ""
            label = f"[{tags}] {title}" if tags else title

            if isinstance(url, str) and url.startswith("http"):
                st.markdown(f"- {label} — [{url}]({url})")
            else:
                st.markdown(f"- {label}")
    else:
        st.caption("No matching policy snippet was found for that question.")

