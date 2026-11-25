import numpy as np

from compare_runs import compare_runs


def test_comparing_a_run_to_itself_has_zero_difference(tmp_path, capsys):
    labels = np.array([0, 1, 2])
    member_probs = np.stack([np.eye(3), np.eye(3), np.eye(3)])
    np.savez(tmp_path / "probs.npz", val_ids=[10, 11, 12], test_ids=[13, 14, 15],
             val_labels=labels, test_labels=labels, val_member_probs=member_probs,
             test_member_probs=member_probs)

    compare_runs(tmp_path, tmp_path, n_boot=10)

    output = capsys.readouterr().out
    assert "Test only (primary)" in output
    assert "Validation + test (secondary)" in output
    assert output.count("Mean B - A +0.0000 | 95% interval [+0.0000, +0.0000]") == 2
    assert output.count("no detectable difference") == 2
