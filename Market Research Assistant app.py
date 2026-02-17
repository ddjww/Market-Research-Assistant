import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ======================================================
# CONFIGURATION and CONSTANTS
# ======================================================
TEMPERATURE = 0.2   
MAX_TOKENS = 800 # cap the model output length 
TOP_K_WIKI = 5 # retrieve top 5 relevant Wikipedia pages
MAX_CHARS_PER_DOC = 6000 # Limit each document to control prompt length and reduce noise.

# ======================================================
# Page config
# ======================================================
st.set_page_config(
    page_title="Market Research Assistant", # browser title
    page_icon="📊", # emoji icon
    layout="wide"  # width layout
)

st.title("WikiPulse — Your Industry Snapshot Generator") # main page title
# Intro text
st.markdown(
    """
     Welcome! This tool helps you quickly build a **500 words industry snapshot** using **Wikipedia** as the only data source.
    **What you will get**
    - A concise industry snapshot powered entirely by Wikipedia.
    - Source transparency, with direct links to the Wikipedia pages used.
    - A structured starting point you can refine, expand, or adapt for deeper analysis.
    """
) 

# ======================================================
# SIDEBAR: Settings (Q0)
# used for configuration inputs (model + API key)
# ======================================================
st.sidebar.header("Settings")

# Dropdown for selecting the LLM 
model_name = st.sidebar.selectbox(
    "Configuration",
    options=["gpt-5"], 
    index=0
)

# Text field for entering API key 
api_key = st.sidebar.text_input(
    "Enter OpenAI API Key",
    type="password",
    help="Your key will not be stored permanently."
)

# ======================================================
# HELPER FUNCTIONS
# ======================================================

# Retrieve the top 5 most relevant Wikipedia pages 
def get_wikipedia_content(industry_query):
    """
    Top 5 relevant Wikipedia pages.
    """
    retriever = WikipediaRetriever(top_k_results=TOP_K_WIKI)
    docs = retriever.invoke(industry_query)
    return docs

# Generate the final industry report using the LLM 
def generate_industry_report(industry, context_text, api_key, model):
    """
    Generate report < 500 words using LLM.
    """
    llm = ChatOpenAI(
        model=model,
        temperature=TEMPERATURE,
        openai_api_key=api_key,
        max_tokens=MAX_TOKENS
    )

    # System-level instructions:
    system_msg = (
        "You are a professional market research analyst writing a concise industry briefing for a business analyst at a large corporation. "
        "CRITICAL EVIDENCE RULE: Use ONLY the provided Wikipedia extracts. "
        "Do not refer to the task, the prompt, or the extracts (avoid phrases like 'the extracts provided' or 'the text does not cover'). "
        "Write in a decision-relevant, analytical tone (not an encyclopedia style). No bullet points."
        "Synthesize information across multiple extracts and ensure every analytical claim is cited."
    )

    # User-level instructions:
    user_msg = f"""
Write a concise and structured industry overview for: {industry}

ABSOLUTE CONSTRAINTS (Zero Tolerance for Deviation):
- Length: 420–450 words (MUST be < 450).
- Structure: EXACTLY 4 long, analytical paragraphs. No headings. No bullet points.
- Tone: Senior Analyst level. Avoid descriptive "encyclopedia" style; use evaluative language.
- Sources: Use ONLY the Wikipedia extracts below. If a claim is not explicitly supported, omit it.
- Integrated Evidence: geography (e.g., Australia, USA) must only appear as short supporting clauses within the flow of your analysis in Paragraphs 3 and 4.
- No meta-language: Do not mention the extracts, the task, or limitations (e.g., avoid “the extracts provided”).

CITATIONS (mandatory):
- Use [Source: Page Title] for key claims.
- Each paragraph must include at least one citation.
- Synthesis: Paragraphs 2, 3, and 4 MUST each blend evidence from 2+ different source pages.
- Do not invent page titles.

PARAGRAPH PLAN (write exactly these 4 paragraphs):
1) Definition & boundary: define what the industry includes (and excludes) as supported by the extracts.
2) Structure & ecosystem: explain key segments/actors AND how they interact (incumbents vs entrants, partnerships), synthesising across sources.
3) Drivers & Dynamics: Analyse the fundamental shifts in demand, delivery, or cost structures.   Regional references (e.g., specific markets) should only serve as brief, high-density evidence for these broader trends, not as standalone descriptions.
4) Critically evaluate the structural risks (e.g., regulation, trust).  Subordinate any regional examples to the analytical argument—they must illustrate a specific friction point, not dominate the narrative.  The paragraph MUST culminate in a sharp, forward-looking analytical implication that accounts for more than 25% of the paragraph's length.

STYLE (secondary):
- No generic conclusion (avoid “In conclusion/Overall…”).

Wikipedia extracts:
{context_text}
""".strip()
    
# Send system + user prompts to the LLM and extract the generated text output.
    report = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]).content

    return report # Return the final report

# ======================================================
# MAIN APP LOGIC
# ======================================================

# Initialize Session State
def init_state():
    # Track which step the user is currently in (1=input, 2=retrieval, 3=report)
    if "steps" not in st.session_state:
        st.session_state.steps = 1
    # Store retrieved Wikipedia documents
    if "docs" not in st.session_state:
        st.session_state.docs = None
    # Store the user’s industry input
    if "industry" not in st.session_state:
        st.session_state.industry = ""
    # Store concatenated Wikipedia context used for generation
    if "wiki_context" not in st.session_state:
        st.session_state.wiki_context = ""

init_state()

# STEP 1: INPUT (Q1)
st.header("Industry Selection")

# Initialize input with session state value
industry_input = st.text_input(
    "Enter an industry to research (e.g., 'Electric Vehicles'):", 
    value=st.session_state.industry
)

# Check if industry is provided
if st.button("Generate"):
    # Validate API key before proceeding
    if not api_key:
        st.error("Please enter your API Key in the sidebar first.")
    # Prevent empty industry queries
    elif not industry_input.strip():
        st.warning("Please enter a industry name.") 
    else:
        st.session_state.industry = industry_input
        st.session_state.steps = 2 # Move workflow to retrieval stage
        # Reset previous retrieval and report state
        st.session_state.docs = None  
        st.session_state.wiki_context = "" 
        # Clear old report if it exists
        if "report_text" in st.session_state:
            st.session_state.report_text = None
            del st.session_state["report_text"] 
        st.rerun() # Force Streamlit rerun

# STEP 2: RETRIEVAL (Q2)
st.divider()
st.header("Data Retrieval")

# If user has not completed Step 1, show placeholder message
if st.session_state.steps < 2:
    st.caption("Sources will appear here after you enter an industry and click Generate.")
else:
    
    if st.session_state.docs is None: # Only trigger retrieval if documents are not already cached in session
        # Display progress status while querying Wikipedia
        with st.status(f"Searching Wikipedia for: {st.session_state.industry}...", expanded=True) as status:
            try:
                raw_docs = get_wikipedia_content(st.session_state.industry)  # Call retriever function

                # Handle case where no relevant pages are returned
                if not raw_docs:
                    status.update(label="No relevant Wikipedia pages found.", state="error", expanded=True)
                    st.error("No relevant Wikipedia pages found. Please try a different industry.")
                    st.stop()

                # Cache top-k documents in session state
                st.session_state.docs = raw_docs[:TOP_K_WIKI]
                status.update(label="Data Retrieval Complete!", state="complete", expanded=False) # Mark retrieval as complete in UI
            except Exception as e:
                # Catch unexpected runtime errors
                status.update(label="Error retrieving data", state="error")
                st.error(f"Error: {e}")
                st.stop()
    
    # Display retrieved sources
    if st.session_state.docs:
        num_docs = len(st.session_state.docs) # Count how many pages were retrieved
        
        # Show warning if fewer than expected sources were found
        if num_docs < TOP_K_WIKI:
            st.warning(
                f"Only {num_docs} relevant Wikipedia pages were found. "
                "The report will be generated based on the available pages."
            )
        else:
            # Confirm successful retrieval
            st.success(f"Found {num_docs} relevant Wikipedia pages.")

        # Build context string that will later be passed to the LLM
        wiki_context = ""
        for i, doc in enumerate(st.session_state.docs):
            source_url = doc.metadata.get("source") # Extract metadata for display and citation
            title = doc.metadata.get("title", "No Title")
            
            # Fallback: construct URL manually if metadata is missing
            if not source_url:
                safe_title = title.replace(" ", "_")
                source_url = f"https://en.wikipedia.org/wiki/{safe_title}"
            
            st.markdown(f"**{i+1}. [{title}]({source_url})**") # Display clickable source link in UI
            
            # Truncate content 
            clean_content = (doc.page_content or "")[:MAX_CHARS_PER_DOC]
            wiki_context += f"Source: {source_url}\nContent: {clean_content}\n\n"
        
        st.session_state.wiki_context = wiki_context

    # If we successfully retrieved docs, we can move to report generation
    # If not, stop here, no point calling the LLM with empty context
    if st.session_state.docs:
        st.session_state.steps = 3
    else:
        st.session_state.steps = 2
        st.stop()

# STEP 3: REPORT (Q3)
st.divider()
st.header("Industry Report")

# user hasn't generated sources yet, so show a placeholder message.
if st.session_state.steps < 3:
    st.caption("Report will appear here after sources are retrieved.")
else:
    #only generate once per run
    # If user clicks Generate again, delete report_text in Step 1 so this block runs again.
    if not st.session_state.get("report_text"):
        with st.spinner("Generating report..."):
            try:
                 # Call the LLM using the user’s industry + the formatted wiki_context
                st.session_state.report_text = generate_industry_report(
                    st.session_state.industry,
                    st.session_state.wiki_context,
                    api_key,
                    model_name
                )
            except Exception as e:
                # If the LLM call fails, show the error and keep report_text empty
                st.error(f"Error generating report: {e}")
                st.session_state.report_text = ""

    # Only display whenactually have text
    if st.session_state.get("report_text"):
        st.write(st.session_state.report_text)