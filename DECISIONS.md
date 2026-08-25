# DECISIONS.md

This file records every place the implementation deviates from the spec and why.

## Phase 0

### D-001: Gemini function-calling uses manual mode (AFC disabled)
**Spec**: Both providers support native function/tool calling.  
**Decision**: For `GeminiClient`, Automatic Function Calling is **disabled** (`AutomaticFunctionCallingConfig(disable=True)`). This keeps the tool-dispatch loop fully in `agent/loop.py` (not hidden inside the SDK), making debugging and tracing transparent.  
**Impact**: None for behavior; same tool-call normalization applies.

### D-002: Groq + Gemini both available; Groq is primary
**User confirmed**: Both API keys present. `LLM_PROVIDER=groq` default. Gemini available as fallback via sidebar dropdown in the Streamlit UI.

### D-003: Selenium scraper is optional / graceful fallback
**Spec**: Refactor Selenium scraper from existing notebook as library function.  
**Decision**: Repository is empty (no notebook to refactor). Implemented `scrape_statement.py` from scratch. Selenium/geckodriver availability is **checked at import time**; if unavailable, `scrape_problem()` returns `None` and logs a warning rather than raising. Primary ingestion uses the Codeforces public API.

## Phase 1

*(to be filled as decisions arise)*

## Phase 2

*(to be filled as decisions arise)*

## Phase 3

*(to be filled as decisions arise)*

## Phase 4

*(to be filled as decisions arise)*

## Phase 5

*(to be filled as decisions arise)*

## Phase 6

*(to be filled as decisions arise)*
