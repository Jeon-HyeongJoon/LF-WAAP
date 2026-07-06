from pathlib import Path

from scripts.msnbc_sequences import iter_sequences, split_records
from waf.infrastructure.behavior.markov_flow import MarkovFlowModel


def test_iter_sequences_skips_headers_and_respects_max_users(tmp_path: Path) -> None:
    data = tmp_path / "msnbc.seq"
    data.write_text(
        "% Different categories found in input file:\n"
        "frontpage news tech\n"
        "% Sequences:\n"
        "1 1\n"
        "2\n"
        "3 2 1\n",
        encoding="utf-8",
    )

    records = list(iter_sequences(data, max_users=2))

    assert [record.states for record in records] == [("frontpage", "frontpage"), ("news",)]


def test_split_records_and_train_markov_on_bounded_sample(tmp_path: Path) -> None:
    data = tmp_path / "msnbc.seq"
    data.write_text(
        "\n".join(
            [
                "% Sequences:",
                "1 2 3",
                "1 2 3",
                "1 2 4",
                "1 2 4",
                "1 2 3",
                "1 2 4",
                "1 2 3",
                "1 2 4",
                "1 2 3",
                "1 2 4",
            ]
        ),
        encoding="utf-8",
    )
    records = list(iter_sequences(data, max_users=10))
    train, validation, test = split_records(records, train_ratio=0.6, validation_ratio=0.2)
    model = MarkovFlowModel(target_fpr=0.2)

    model.fit([record.states for record in train], [record.states for record in validation])

    assert model.is_trained
    assert len(train) == 6
    assert len(validation) == 2
    assert len(test) == 2
    assert not model.assess_transition("frontpage", "news").blocked
