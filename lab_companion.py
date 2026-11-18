"""Small, explainable ML + retrieval + deterministic tool-routing functions."""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

TRAINING = [
    ("what is the aim of the experiment", "document"), ("explain the theory in the guide", "document"),
    ("where does the handout describe apparatus", "document"), ("summarize the procedure", "document"),
    ("ask me viva questions about the experiment", "viva"), ("quiz me on the theory", "viva"),
    ("practice oral exam questions", "viva"), ("give me a viva on safety", "viva"),
    ("calculate current from voltage and resistance", "calculation"), ("what is 5 volts divided by 100 ohms", "calculation"),
    ("compute the percentage error", "calculation"), ("find the resistance using ohms law", "calculation"),
    ("is it safe to touch the circuit", "safety"), ("what precautions should I follow", "safety"),
    ("can I connect this while power is on", "safety"), ("how do I avoid electric shock", "safety"),
]
_train_texts, _train_labels = zip(*TRAINING)
_intent_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
_intent_model = LogisticRegression(max_iter=500, random_state=7).fit(_intent_vectorizer.fit_transform(_train_texts), _train_labels)


def classify_intent(text: str) -> str:
    """Return a best-effort intent from the tiny demonstration classifier."""
    if re.search(r"\b(safe|safety|shock|danger|precaution|power on)\b", text.lower()):
        return "safety"  # safety requests always go to the cautionary handler
    return str(_intent_model.predict(_intent_vectorizer.transform([text]))[0])


def route_request(intent: str) -> str:
    return {"document": "document_qa", "viva": "viva_generator", "calculation": "calculator", "safety": "safety_guidance"}.get(intent, "document_qa")


def _chunks(text: str):
    """Split Markdown and lab-manual text into passages with useful source labels."""
    chunks = []
    experiment = "General"
    section = "Overview"
    buffer = []

    def flush():
        body = " ".join(buffer).strip()
        if body:
            source = f"{experiment} — {section}" if experiment != "General" else section
            chunks.append((source, f"{source}. {body}"))
        buffer.clear()

    experiment_re = re.compile(r"^EXPERIMENT\s*[–—-]?\s*(\d+(?:\.\d+)*)\s*$", re.IGNORECASE)
    section_re = re.compile(
        r"^(Aim|Objective|Theory|Apparatus|Material\s+Required|Materials?\s+Required|"
        r"Formulae?\s+Used|Observations?|Observation\s+Table|Calculations?|"
        r"Procedure|Discussion\s+Questions?|Result|Precautions|Sources?\s+of\s+Error)\s*:?\s*(.*)$",
        re.IGNORECASE,
    )

    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if re.fullmatch(r"Page \d+ \(PHYSICS\)", line, re.IGNORECASE):
            continue

        experiment_match = experiment_re.match(line)
        if experiment_match:
            flush()
            experiment = f"Experiment {experiment_match.group(1)}"
            section = "Overview"
            continue

        markdown_heading = re.match(r"^#{1,3}\s+(.+)$", line)
        if markdown_heading:
            flush()
            section = markdown_heading.group(1).strip()
            if section.lower().startswith("document:"):
                experiment = "General"
            continue

        section_match = section_re.match(line)
        if section_match:
            flush()
            section = section_match.group(1).strip().title()
            inline_text = section_match.group(2).strip()
            if inline_text:
                buffer.append(inline_text)
            continue

        buffer.append(line)

    flush()
    return chunks


def retrieve(query: str, document: str, limit: int = 3):
    query = re.sub(r"\bformulae\b", "formula", query, flags=re.IGNORECASE)
    chunks = _chunks(document)
    if not chunks:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
    texts = [re.sub(r"\bformulae\b", "formula", chunk[1], flags=re.IGNORECASE) for chunk in chunks]
    matrix = vectorizer.fit_transform(texts + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    requested_experiment = re.search(r"\bexperiment\s+(\d+(?:\.\d+)*)\b", query, flags=re.IGNORECASE)
    candidate_indices = []
    if requested_experiment:
        prefix = f"Experiment {requested_experiment.group(1)} —".casefold()
        candidate_indices = [i for i, (source, _) in enumerate(chunks) if source.casefold().startswith(prefix)]

    # Give explicitly requested manual sections priority over general TF-IDF overlap.
    # This matters for short section names: "procedure" may otherwise rank Aim/Result higher.
    section_patterns = [
        (r"\b(aim|objective)\b", {"aim", "objective"}),
        (r"\b(procedure|method|steps)\b", {"procedure"}),
        (r"\b(apparatus|materials? required)\b", {"apparatus", "material required", "materials required"}),
        (r"\b(formula|formulae|equation)\b", {"formula used", "formulae used"}),
        (r"\b(precaution|precautions|safety)\b", {"precautions"}),
        (r"\b(result|conclusion)\b", {"result"}),
        (r"\b(theory|principle)\b", {"theory"}),
        (r"\b(observation|observations)\b", {"observation", "observations", "observation table"}),
        (r"\b(calculation|calculations)\b", {"calculation", "calculations"}),
    ]
    requested_sections = next((labels for pattern, labels in section_patterns if re.search(pattern, query, re.IGNORECASE)), None)
    section_candidates = candidate_indices if candidate_indices else list(range(len(chunks)))
    if requested_sections:
        matching_sections = [
            i for i in section_candidates
            if chunks[i][0].rsplit(" — ", 1)[-1].casefold() in requested_sections
        ]
        if matching_sections:
            candidate_indices = matching_sections

    order = sorted(candidate_indices, key=lambda i: scores[i], reverse=True) if candidate_indices else scores.argsort()[::-1].tolist()
    order = order[:limit]
    return [{"source": chunks[i][0], "text": chunks[i][1], "score": float(scores[i])} for i in order if scores[i] > 0]


def _safety_text():
    return "Safety guidance is not a substitute for your instructor or the approved lab manual. Do not energize, modify, or touch a circuit unless your instructor has approved the setup and you have been trained. Isolate power before changing connections; follow local PPE and equipment rules. Ask the instructor to confirm any uncertainty."


def answer_question(question: str, document: str):
    intent = classify_intent(question)
    if intent == "safety":
        return {"answer": _safety_text(), "sources": [], "warning": "Safety response is intentionally conservative and is not a procedure."}
    if intent == "calculation":
        values = [float(x) for x in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", question)]
        if len(values) >= 2 and re.search(r"\b(current|amps?|amperes?)\b", question.lower()):
            volts, ohms = values[0], values[1]
            if ohms == 0:
                return {"answer": "Resistance cannot be zero for this calculation.", "sources": [], "warning": None}
            return {"answer": f"Using Ohm's law, I = V/R = {volts:g} V / {ohms:g} Ω = {volts / ohms:.4g} A. Check units and confirm the values with your instructor.", "sources": [], "warning": None}
        return {"answer": "The demo calculator only handles current when a question includes voltage and resistance as numbers (for example: 'calculate current from 5 V and 100 ohms'). Check units before using a result.", "sources": [], "warning": None}
    if re.search(r"\bprocedure\b", question, flags=re.IGNORECASE):
        requested = re.search(r"\bexperiment\s+(\d+(?:\.\d+)*)\b", question, flags=re.IGNORECASE)
        if requested:
            exp_name = f"Experiment {requested.group(1)}"
            labels = [source.split(" — ", 1)[1] for source, _ in _chunks(document) if source.startswith(exp_name + " — ")]
            has_procedure = any(re.search(r"procedure|method|steps", label, flags=re.IGNORECASE) for label in labels)
            if labels and not has_procedure:
                available = ", ".join(labels)
                return {"answer": f"The document has no labeled procedure section for {exp_name}. It includes: {available}.", "sources": [], "warning": "Use the instructor-approved procedure; this app will not infer missing lab steps."}
    sources = retrieve(question, document)
    explicit_procedure_match = bool(
        re.search(r"\b(procedure|method|steps)\b", question, flags=re.IGNORECASE)
        and sources
        and sources[0]["source"].casefold().endswith(" — procedure")
    )
    if not sources or (sources[0]["score"] < 0.08 and not explicit_procedure_match):
        return {"answer": "I could not find a close match in the active handout. Try using a term from a section heading, or add the relevant approved lab text.", "sources": sources, "warning": None}
    return {"answer": " ".join(s["text"] for s in sources[:2]), "sources": sources[:2], "warning": "This is an extractive answer from the uploaded text. Check the cited section; the app does not verify whether the document itself is correct."}


VIVA_BANK = [
    (("ohm", "experiment 1", "resistance per cm"), "Built-in bank — Experiment 1: Ohm's law", [
        "State Ohm's law and name the condition under which it applies.",
        "What does the slope of a potential-difference versus current graph represent?",
        "Why is an ammeter connected in series and a voltmeter in parallel?",
        "How can the resistance per centimetre of a wire be estimated from its measured resistance and length?",
        "Name two possible sources of error when measuring voltage and current.",
    ]),
    (("wheatstone", "whetstone", "meter bridge", "experiment 2"), "Built-in bank — Experiment 2: Meter bridge", [
        "What is the principle of a meter bridge?",
        "What is the balance point and what is the galvanometer reading there?",
        "Why should a meter-bridge wire be uniform in thickness and material?",
        "Why should the jockey touch the bridge wire gently?",
        "How does interchanging the known and unknown resistances help check a result?",
    ]),
    (("series", "parallel", "combination of resistance", "experiment 3"), "Built-in bank — Experiment 3: Series and parallel resistance", [
        "How does the equivalent resistance of two resistors in series compare with each individual resistance?",
        "How does the equivalent resistance of parallel resistors compare with the smallest branch resistance?",
        "How are voltage and current distributed in series and parallel circuits?",
        "Why compare experimental and theoretical equivalent resistance?",
        "Name one reason measured resistance may differ from the theoretical value.",
    ]),
    (("galvanometer", "half deflection", "figure of merit", "experiment 4"), "Built-in bank — Experiment 4: Galvanometer", [
        "What is the half-deflection method used to determine?",
        "What does the figure of merit of a galvanometer represent, and what are its units?",
        "What is the role of a shunt in the half-deflection method?",
        "Why is a high series resistance introduced initially in this experiment?",
        "Which readings are needed to calculate the galvanometer's figure of merit?",
    ]),
    (("concave mirror", "experiment 5"), "Built-in bank — Experiment 5: Concave mirror", [
        "State the mirror formula and identify its quantities.",
        "How can focal length be estimated from an object-distance and image-distance graph?",
        "What is parallax and how is it removed when locating an image?",
        "Why should the optical bench uprights be vertical and aligned?",
    ]),
    (("convex mirror", "experiment 6"), "Built-in bank — Experiment 6: Convex mirror", [
        "How is the focal length of a convex mirror related to its radius of curvature?",
        "Why is a convex lens used with the convex mirror in this method?",
        "What alignment checks are important on the optical bench?",
    ]),
    (("convex lens", "experiment 7"), "Built-in bank — Experiment 7: Convex lens", [
        "State the thin-lens formula used to relate object distance, image distance, and focal length.",
        "How can focal length be found from a graph of reciprocal distances?",
        "What does removing parallax confirm in an optical-bench measurement?",
        "Why should the object and image needle tips be at the lens-centre height?",
    ]),
    (("concave lens", "experiment 8"), "Built-in bank — Experiment 8: Concave lens", [
        "Why is a convex lens combined with the concave lens in this method?",
        "What sign is associated with the focal length of a concave lens under the usual sign convention?",
        "Why should the lenses be clean and aligned on the optical bench?",
    ]),
    (("prism", "minimum deviation", "refractive index", "experiment 9"), "Built-in bank — Experiment 9: Prism", [
        "What is the angle of minimum deviation?",
        "How does the deviation angle change as incidence angle approaches the minimum-deviation condition?",
        "How is a prism's refractive index related to its prism angle and minimum deviation?",
        "Why should the pins be vertical and spaced apart during the prism experiment?",
    ]),
]


def generate_viva(document: str, topic: str = "", count: int = 5):
    chunks = _chunks(document)
    if topic.strip():
        hits = retrieve(topic, document, limit=max(count, 3))
        selected = [(h["source"], h["text"]) for h in hits] or chunks
    else:
        selected = chunks
    if not selected:
        return [{"question": "Upload a lab guide to generate questions.", "reason": "No source text is available.", "source": "None", "hint": ""}]

    explicit_questions = []
    seen_questions = set()
    for source, body in selected:
        for question_text in re.findall(r"[^.!?\n]{8,220}\?", body):
            question_text = question_text.strip(" \t-•0123456789.)")
            if question_text and question_text.casefold() not in seen_questions:
                seen_questions.add(question_text.casefold())
                explicit_questions.append({
                    "question": question_text,
                    "reason": "Taken from an uploaded viva or reference document.",
                    "source": source,
                    "hint": "Try answering from your uploaded materials, then confirm with your instructor.",
                })

    search_text = (topic + " " + " ".join(source for source, _ in selected)).casefold()
    offline_questions = []
    for tags, source, questions in VIVA_BANK:
        if any(tag in search_text for tag in tags):
            offline_questions.extend({
                "question": question,
                "reason": "Built-in offline practice question; not fetched live from the web.",
                "source": source,
                "hint": "Use the matching section in your practical PDF and verify details with your instructor.",
            } for question in questions)

    templates = [
        ("What is the main idea in '{heading}'?", "Checks understanding of the source section."),
        ("Which evidence or observation would support the explanation in '{heading}'?", "Connects theory with measurable observations."),
        ("What assumption or limitation should you check in '{heading}'?", "Encourages critical thinking about the handout."),
        ("How would you explain '{heading}' in your own words?", "Checks conceptual understanding rather than memorization."),
        ("What could cause a result related to '{heading}' to differ from expectation?", "Prompts analysis of experimental error."),
        ("What should you verify before applying '{heading}' in the lab?", "Prompts safe preparation and source checking."),
        ("How could you test whether your interpretation of '{heading}' is correct?", "Connects the section to validation."),
        ("What is one question you would ask your instructor about '{heading}'?", "Encourages appropriate escalation of uncertainty."),
    ]
    results = explicit_questions[:count]
    seen_questions.update(item["question"].casefold() for item in results)
    for item in offline_questions:
        if len(results) >= count:
            break
        if item["question"].casefold() not in seen_questions:
            results.append(item)
            seen_questions.add(item["question"].casefold())
    template_sources = [(source, text) for source, text in selected if not re.search(r"viva|question bank", source, flags=re.IGNORECASE)] or selected
    for i in range(len(results), count):
        offset = i - len(explicit_questions) - len(offline_questions)
        heading, text = template_sources[offset % len(template_sources)]
        question, reason = templates[offset % len(templates)]
        hint = re.sub(r"\s+", " ", text)[:230]
        results.append({"question": question.format(heading=heading), "reason": reason, "source": heading, "hint": f"Look for: {hint}"})
    return results
