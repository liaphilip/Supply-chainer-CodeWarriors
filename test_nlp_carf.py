import sys
import os

# Ensure backend imports work
sys.path.insert(0, os.getcwd())

from backend.engine.threat_intelligence import CARFFilter, ContrastiveNLPEngine

def test_carf_punctuation_and_normalization():
    print("\n--- TEST: CARF Text Normalization & Punctuation ---")
    carf = CARFFilter()

    # Attached punctuation tests
    assert carf.apply_filter(0.8, "Operations halted at the primary port,", "sea") == 0.8, "Failed: 'port,'"
    assert carf.apply_filter(0.8, "Disruption reported at international terminal.", "air") == 0.8, "Failed: 'terminal.'"
    assert carf.apply_filter(0.8, "Heavy congestion on connecting road,", "road") == 0.8, "Failed: 'road,'"
    assert carf.apply_filter(0.8, "Service suspended on regional railway.", "rail") == 0.8, "Failed: 'railway.'"

    # Case insensitivity tests
    assert carf.apply_filter(0.75, "CRITICAL: CONTAINER SHIP DELAYED AT PORT", "SEA") == 0.75, "Failed uppercase SEA mode"
    assert carf.apply_filter(0.75, "FLIGHT CANCELLED AT AIRPORT", "Air") == 0.75, "Failed mixed case Air mode"

    print("PASS: CARF Text Normalization & Punctuation")

def test_carf_four_transport_modes():
    print("\n--- TEST: CARF Modality Filtering Across All 4 Modes ---")
    carf = CARFFilter()
    score = 0.85

    # 1. SEA Tests
    sea_relevant = "Container vessel ran aground in Suez Canal."
    sea_irrelevant = "Major flight cancellations at Heathrow airport."
    assert carf.apply_filter(score, sea_relevant, "sea") == score, "SEA relevant failed"
    assert carf.apply_filter(score, sea_irrelevant, "sea") == 0.0, "SEA irrelevant failed"

    # 2. AIR Tests
    air_relevant = "Aviation strike grounding all flights at international airport."
    air_irrelevant = "Port workers strike freezing container ship movements."
    assert carf.apply_filter(score, air_relevant, "air") == score, "AIR relevant failed"
    assert carf.apply_filter(score, air_irrelevant, "air") == 0.0, "AIR irrelevant failed"

    # 3. ROAD Tests
    road_relevant = "Highway accident blocking truck delivery corridor."
    road_irrelevant = "Vessel stranded in ocean port."
    assert carf.apply_filter(score, road_relevant, "road") == score, "ROAD relevant failed"
    assert carf.apply_filter(score, road_irrelevant, "road") == 0.0, "ROAD irrelevant failed"

    # 4. RAIL Tests
    rail_relevant = "Train derailment damaging main railway track."
    rail_irrelevant = "Airport runway maintenance delaying flights."
    assert carf.apply_filter(score, rail_relevant, "rail") == score, "RAIL relevant failed"
    assert carf.apply_filter(score, rail_irrelevant, "rail") == 0.0, "RAIL irrelevant failed"

    print("PASS: CARF Modality Filtering Across All 4 Modes")

def test_nlp_regression_behavior():
    print("\n--- TEST: NLP Contrastive Engine Regression ---")
    nlp = ContrastiveNLPEngine()
    nlp.warmup()

    # 1. Safe news text
    safe_text = "Clear weather, normal operations, traffic is flowing smoothly."
    safe_score = nlp.get_semantic_score(safe_text)
    assert safe_score == 0.0, f"Expected 0.0 for safe text, got {safe_score}"

    # 2. Genuine disruption
    disrupt_text = "Catastrophic hurricane destroys major container port and blocks maritime channels."
    disrupt_score = nlp.get_semantic_score(disrupt_text)
    assert disrupt_score > 0.0, f"Expected > 0.0 for disruption, got {disrupt_score}"

    # 3. Range assertions
    test_texts = [
        safe_text,
        disrupt_text,
        "Minor berthing congestion at terminal B-4.",
        "",
        "   "
    ]
    for text in test_texts:
        sc = nlp.get_semantic_score(text)
        assert 0.0 <= sc <= 1.0, f"Score out of range [0, 1]: {sc} for '{text}'"

    print("PASS: NLP Contrastive Engine Regression")

if __name__ == "__main__":
    test_carf_punctuation_and_normalization()
    test_carf_four_transport_modes()
    test_nlp_regression_behavior()
    print("\nALL TEAM 1 NLP & CARF TESTS PASSED SUCCESSFULLY.")
