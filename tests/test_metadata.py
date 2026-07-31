from code_grpo.metadata import experiment_metadata


def test_metadata_is_serializable_and_contains_provenance():
    metadata = experiment_metadata()
    assert metadata["python"]
    assert metadata["created_at_utc"].endswith("+00:00")
    assert "transformers" in metadata["packages"]
