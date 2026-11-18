# AI-Powered Engineering Lab Companion

A beginner-friendly, offline-first demonstration app for a Windows laptop. It combines a small scikit-learn intent classifier, TF-IDF document retrieval, built-in offline viva practice, an optional Gemini API viva-question generator, a tiny Ohm's law calculator, and a deterministic tool router. Document Q&A and offline viva questions need no API key. Gemini generation needs internet access and your own Google AI Studio API key.

> **Curriculum status:** The 5-Day Skill Development Program Structure PDF is mapped in program_alignment.md. The assessment brief was not supplied, so exact rubric alignment is pending. This is intentionally a mini-demo, not a complete implementation of the five-day curriculum. The included handout is sample content.

## Run on Windows

1. Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/windows/). During setup, select **Add Python to PATH**.
2. Open PowerShell in this project folder.
3. Create an isolated environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that same window, then activate again. This affects only the current PowerShell process.

4. Install the dependencies (internet required for this one-time step):

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Launch:

   ```powershell
   streamlit run app.py
   ```

6. Streamlit opens the app in a browser, usually at `http://localhost:8501`. Press `Ctrl+C` in PowerShell to stop it.

## Add outside viva questions

The app includes a built-in offline question bank for the experiments in the practical PDF, so you do not need a separate viva file. Enter a topic such as Whetstone bridge, Experiment 4, or prism. You may optionally upload another reference/question file too; explicit questions in uploaded sources are shown first. Offline mode fetches nothing. Optional Gemini mode creates real four-option viva MCQs from broad course knowledge. You may optionally send relevant excerpts from your uploaded manual as extra context; this is not a live Google or web search.
## Demo flow (3–5 minutes)

1. Ask: `What is the aim of the experiment?` Show the predicted intent, selected tool, retrieved evidence, and section citation.
2. Ask: `Calculate current from 5 V and 100 ohms.` The local calculator returns 0.05 A using `I = V/R`.
3. Ask a safety question. The conservative safety handler explains that it cannot approve a setup and directs the learner to the instructor.
4. Generate five viva questions for `Ohm's law`. Explain that retrieval selects relevant sections, and fixed templates create questions tied to those sources.
5. Upload one or more approved source files in the sidebar. You can select your practical PDF and a separate viva-question file together. Supported: .txt, .md, .pdf, .docx.

## What's inside

- `app.py`: Streamlit interface, document upload, and optional API controls.
- `lab_companion.py`: scikit-learn intent model, deterministic routing, TF-IDF retrieval, extractive Q&A, offline viva bank/templates, and constrained calculator.
- `gemini_viva.py`: optional Gemini API integration, using only the retrieved excerpts supplied from the uploaded document.
- `data/sample_lab_guide.txt`: fictional sample source, explicitly not lab instructions.
- `presentation_outline.md`: slide-by-slide presentation outline.
- `report_outline.md`: short report scaffold.

## Mock viva multiple-choice practice

Generate questions with Gemini enabled for a standard viva quiz: each question has four answer choices, one correct answer, and an explanation. Gemini uses broad course knowledge; you can optionally include matching uploaded-manual text as extra context. Without Gemini, offline practice stays available and asks you to choose the manual statement that best supports an answer. Both modes track score, show progress, and allow a retry. AI answers and retrieval matches can be wrong, so verify them with your teacher or textbook.

## Mini-demo scope

For a short class presentation, show only three flows: ask a question about the sample guide and show the retrieved section, calculate current from 5 V and 100 ohms, and generate a few viva prompts. You do not need to build neural networks, connect an LLM API, or implement multi-agent coordination for this demonstration. Be clear that these are outside the current prototype.
## Design and limitations

- Intent classification uses a tiny hand-written training set. It is a learning demonstration, not a reliable general classifier. Safety-related words are routed conservatively.
- Retrieval is lexical TF-IDF, not a semantic embedding model. It can miss paraphrases and may surface irrelevant text.
- Q&A returns source extracts rather than generating new claims. It cannot check source accuracy, completeness, or instructor approval.
- Offline viva questions use a built-in bank and deterministic templates. If enabled, Gemini generates four-choice questions from broad course knowledge; uploaded-document excerpts are optional extra context. The “agentic” behavior is a transparent sequence of classify → route → retrieve/generate/calculate → show evidence.
- The calculator handles only a limited current calculation using numbers in the prompt. It is not suitable for design or real circuit setup.
- PDFs are read with pypdf; scanned/image-only PDFs cannot be read without OCR, and equation layouts may be distorted. Word .docx text is extracted locally. Verify equations in the original document. Gemini mode sends the topic and matching uploaded-document excerpts to Google. Google states that free-tier submitted content may be used to improve its products. Never submit private or confidential documents. The key is entered in a password field and is not written into project files. Do not enable billing if you want to keep this project at ₹0; free quota availability and limits can change, and requests may be blocked when limits are reached.
- Use approved materials, confirm all results with your instructor, and follow equipment manuals and local lab safety rules. Never use this demo to authorize a real-world experiment.

## Optional Gemini API viva generation

1. Create your own Gemini API key in Google AI Studio. Do not post the key in chat, screenshots, or source files.
2. In the viva coach, select **Use Gemini API**, enter the key in the password field, optionally check **Remember this key on this computer**, and agree to the data-sharing note. The key is saved only when you click **Generate viva questions**. The saved key file is local, unencrypted, and excluded from the project ZIP; do not share the project folder while it is saved. Use **Forget saved API key** to remove it.
3. Enter a topic (for example, “meter bridge” or “enthalpy of neutralisation”). Gemini can use general course knowledge even when the topic is not in your PDF. Check “Use my uploaded manual as extra context” only when you want to send relevant excerpts too; the app asks you to confirm before sending them.
4. Gemini generates four answer choices, a correct-answer index, and a short explanation. It does not browse the live web, and its answers may be wrong. To stay at ₹0, do not enable billing. Free quota availability and limits can change; offline practice remains available without a key.






