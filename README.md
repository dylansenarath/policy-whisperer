\# Policy Whisperer — Demo



Policy-grounded answers for clinicians, with inline citations and source links.  

\*\*Not medical advice.\*\* Demo corpus uses 3 sample PDFs.



\*\*Live app:\*\* <ADD\_YOUR\_STREAMLIT\_URL\_HERE>



\## What it does

\- Answers questions \*\*only\*\* from your policies (RAG).

\- Shows inline citations like `\[S1]` and a \*\*Sources\*\* list you can click.

\- Corpus is \*\*company-controlled\*\*: PDFs live in `data/pdfs/`.

\- Persists a local vector index (Chroma) for fast repeat queries.



\## Demo corpus (included)

\- `hand\_hygiene.pdf`

\- `insulin\_administration.pdf`

\- `restraints\_medical\_behavioral.pdf`



\## Quick start (local)

```bash

python -m venv .venv \&\& .venv\\Scripts\\activate    # on Windows

pip install -r requirements.txt

set GROQ\_API\_KEY=YOUR\_KEY

streamlit run streamlit\_app.py



\## Deploy (Streamlit Community Cloud)

Main file: streamlit_app.py

Secrets:

GROQ_API_KEY = "YOUR_KEY"
GROQ_MODEL   = "llama-3.3-70b-versatile"
USER_AGENT   = "PolicyWhisperer/0.1 (+https://example.org)"



\## Customize the corpus

Put PDFs in data/pdfs/ and list them in rag_core.py → PDF_PATHS.
You can also add stable URLs in SEED_URLS.



\## Safety

Outputs summarize policy documents and may be incomplete or outdated.
Always verify against official policies/protocols and clinical judgment.
