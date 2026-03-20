import re

def clean_text(text):
    """
    Cleans extracted PDF text by removing repetitive headers/footers 
    and normalizing whitespace.
    """
    # Example pattern for the university headers seen in the notebook
    header_pattern = r"CNPN\s*_\s*M_MS\s*_\s*2014\s*/\s*Version\s*_\s*SGG1"
    text = re.sub(header_pattern, "", text)
    text = re.sub(r"\n\s*\n", "\n", text)  # Remove multiple newlines
    return text.strip()
