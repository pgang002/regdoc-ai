from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_document_benchmark_has_source_group_separation() -> None:
    manifest = pd.read_csv(ROOT / "data/processed/document_understanding/manifest.csv")
    train_groups = set(manifest.loc[manifest["split"] == "train", "source_group"])
    test_groups = set(manifest.loc[manifest["split"] == "test", "source_group"])
    assert train_groups == {"NCT04470427"}
    assert test_groups == {"NCT04796896"}
    assert train_groups.isdisjoint(test_groups)


def test_document_benchmark_contains_all_five_classes() -> None:
    manifest = pd.read_csv(ROOT / "data/processed/document_understanding/manifest.csv")
    expected = {"FDA_1572", "FDA_3454", "FDA_3455", "CLINICAL_PROTOCOL", "CLINICAL_TABLE"}
    assert set(manifest["class_label"]) == expected
    assert manifest["actual_content"].all()
    for relative in manifest["image_path"]:
        assert (ROOT / relative).exists()
