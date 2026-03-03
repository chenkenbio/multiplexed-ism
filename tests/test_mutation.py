"""Tests for mutation.py: ALT_TABLE, sampling, and apply_mutations."""

import torch
import pytest

from multiplexed_ism.mutation import (
    ALT_TABLE,
    apply_mutations,
    get_ref_bases,
    sample_positions,
)


class TestAltTable:
    def test_shape(self) -> None:
        assert ALT_TABLE.shape == (4, 3)

    def test_excludes_self(self) -> None:
        for ref in range(4):
            alts = ALT_TABLE[ref].tolist()
            assert ref not in alts

    def test_sorted(self) -> None:
        for ref in range(4):
            alts = ALT_TABLE[ref].tolist()
            assert alts == sorted(alts)

    def test_covers_all_alts(self) -> None:
        for ref in range(4):
            alts = set(ALT_TABLE[ref].tolist())
            expected = set(range(4)) - {ref}
            assert alts == expected


class TestGetRefBases:
    def test_basic(self, short_seq: torch.Tensor) -> None:
        ref = get_ref_bases(short_seq, 0, 20)
        assert ref.shape == (20,)
        assert torch.all((ref >= 0) & (ref <= 3))
        # ref should match the argmax of the one-hot
        expected = short_seq.argmax(dim=0)
        assert torch.equal(ref, expected)

    def test_subregion(self, short_seq: torch.Tensor) -> None:
        ref = get_ref_bases(short_seq, 5, 15)
        assert ref.shape == (10,)
        expected = short_seq[:, 5:15].argmax(dim=0)
        assert torch.equal(ref, expected)


class TestSamplePositions:
    def test_shape(self) -> None:
        pos = sample_positions(100, 10, 50)
        assert pos.shape == (50, 10)

    def test_no_replacement_per_row(self) -> None:
        pos = sample_positions(20, 5, 100)
        for i in range(100):
            row = pos[i].tolist()
            assert len(row) == len(set(row)), f"Duplicate in row {i}"

    def test_in_range(self) -> None:
        L_mut = 30
        pos = sample_positions(L_mut, 10, 50)
        assert torch.all(pos >= 0)
        assert torch.all(pos < L_mut)

    def test_seed_reproducibility(self) -> None:
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        pos1 = sample_positions(50, 5, 20, gen1)
        pos2 = sample_positions(50, 5, 20, gen2)
        assert torch.equal(pos1, pos2)

    def test_n_var_equals_L(self) -> None:
        """When n_var_per_run == L_mut, every position is selected."""
        pos = sample_positions(10, 10, 5)
        for i in range(5):
            assert set(pos[i].tolist()) == set(range(10))


class TestApplyMutations:
    def test_one_hot_preserved(self, short_seq: torch.Tensor) -> None:
        ref_bases = get_ref_bases(short_seq, 0, 20)
        pos = torch.tensor([3, 7, 12])
        mutated = apply_mutations(short_seq, ref_bases, pos, alt_idx=0, start=0)

        # Check one-hot: each column sums to 1
        col_sums = mutated.sum(dim=0)
        assert torch.allclose(col_sums, torch.ones(20))

    def test_correct_alt_base(self, short_seq: torch.Tensor) -> None:
        ref_bases = get_ref_bases(short_seq, 0, 20)
        pos = torch.tensor([5])
        ref_at_5 = ref_bases[5].item()
        expected_alt = ALT_TABLE[ref_at_5, 1].item()

        mutated = apply_mutations(short_seq, ref_bases, pos, alt_idx=1, start=0)
        assert mutated[expected_alt, 5].item() == 1.0
        assert mutated[ref_at_5, 5].item() == 0.0

    def test_unmutated_positions_unchanged(self, short_seq: torch.Tensor) -> None:
        ref_bases = get_ref_bases(short_seq, 0, 20)
        pos = torch.tensor([3])
        mutated = apply_mutations(short_seq, ref_bases, pos, alt_idx=0, start=0)

        # All positions except 3 should be unchanged
        for p in range(20):
            if p != 3:
                assert torch.equal(mutated[:, p], short_seq[:, p])

    def test_original_unchanged(self, short_seq: torch.Tensor) -> None:
        original = short_seq.clone()
        ref_bases = get_ref_bases(short_seq, 0, 20)
        pos = torch.tensor([0, 5, 10])
        apply_mutations(short_seq, ref_bases, pos, alt_idx=2, start=0)
        assert torch.equal(short_seq, original)

    def test_with_offset(self, short_seq: torch.Tensor) -> None:
        start = 5
        ref_bases = get_ref_bases(short_seq, start, 15)
        pos = torch.tensor([0])  # position 0 in mutable region = position 5 in seq
        mutated = apply_mutations(short_seq, ref_bases, pos, alt_idx=0, start=start)

        # Position 5 in the sequence should be mutated
        ref_at_5 = short_seq.argmax(dim=0)[5].item()
        expected_alt = ALT_TABLE[ref_at_5, 0].item()
        assert mutated[expected_alt, 5].item() == 1.0
