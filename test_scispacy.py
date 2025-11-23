try:
    import spacy  # type: ignore[import]
except ImportError:
    raise SystemExit(
        "Missing dependency: 'spacy'. Install with:\n"
        "  pip install spacy scispacy\n"
        "Then install the scispaCy model, for example:\n"
        "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz"
    )

# Load small scispaCy medical model
MODEL = "en_core_sci_sm"
try:
    nlp = spacy.load(MODEL)
except OSError:
    raise SystemExit(
        f"Model '{MODEL}' is not installed. Install it with:\n"
        f"  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/{MODEL}-0.5.1.tar.gz"
    )

# Sample medical text
text = "The patient was diagnosed with hypertension and prescribed Metformin."

doc = nlp(text)

print("Detected entities:")
for ent in doc.ents:
    print(ent.text, "-", ent.label_)