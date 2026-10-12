"""Check semantic tie targets and all-denominator preference evaluation."""

import sys

from pathlib import Path

import pytest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))

from train_tie_script_comparator import genuine_tie, target_distribution, swap_options, metrics, opaque_member, train_option_order, weighted_training_loss, length_adaptive_groups, verify_batching_continuation

from train_contextual_script_comparator import padded_pair_inputs

def test_split_votes_do_not_become_indifference():
    row={'preference_probability':.5,'original_winner':'tie','original_signed_strengths':[-1,1]}
    assert not genuine_tie(row)
    assert target_distribution(row,.1)==[.5,.5,0.]

def test_original_human_indifference_is_preserved():
    row={'preference_probability':.5,'winner':'tie'}
    assert genuine_tie(row)
    assert target_distribution(row,.1)==[0.,0.,1.]

def test_individual_indifference_and_disagreement_remain_separate():
    row={'preference_probability':.5,'original_winner':'tie','original_signed_strengths':[-1,0,1]}
    assert not genuine_tie(row)
    assert target_distribution(row)==[1/3,1/3,1/3]
    row['preference_probability']=1
    with pytest.raises(ValueError):target_distribution(row)

def test_actual_two_tie_votes_required():
    row={'preference_probability':.5,'synthetic':True,'reviews':{}}
    with pytest.raises(ValueError):target_distribution(row,.1)
    row['reviews']={f'gpt-6-luna:{order}':{'winner':'tie','form_a':True,'form_b':True} for order in ('AB','BA')}
    assert target_distribution(row,.1)==[.05,.05,.9]
    row['reviews']['gpt-6-luna:BA']['winner']='B'
    with pytest.raises(ValueError):target_distribution(row)

def test_softening_preserves_human_targets_and_old_binary_policy():
    assert target_distribution({'preference_probability':1,'synthetic':True},.1)==[.9,.1,0.]
    assert target_distribution({'preference_probability':.8},.1)==[.8,.19999999999999996,0.]

def test_symmetry_preserves_tie_probability():
    ab=torch.tensor([2.,-1.,.4]);ba=torch.tensor([-.6,1.3,.2])
    p=((ab+swap_options(ba))/2).softmax(-1)
    reversed_p=((ba+swap_options(ab))/2).softmax(-1)
    assert torch.allclose(p,swap_options(reversed_p))

def test_false_ties_count_as_errors_on_human_choices():
    report=metrics([{'human_label':'A','prediction':'tie'}, {'human_label':'tie','prediction':'tie'},
                    {'human_label':'B','prediction':'B'}])
    assert report['three_way_correct']==2
    assert report['decisive_accuracy']==.5
    assert report['tie_precision']==.5
    assert report['false_ties_on_decisive']==1

def test_native_pointer_padding_keeps_complete_inputs_and_original_positions():
    items = [{'input_ids': torch.tensor([[4, 5, 6, 7]]), 'attention_mask': torch.ones(1, 4, dtype=torch.long),
              'opt_idx': torch.tensor([[1, 2, 3]]), 'n_slots': torch.tensor([3])},
             {'input_ids': torch.tensor([[8, 9, 10, 11, 12, 13]]), 'attention_mask': torch.ones(1, 6, dtype=torch.long),
              'opt_idx': torch.tensor([[2, 4, 5]]), 'n_slots': torch.tensor([3])}]
    batch = padded_pair_inputs(items, 0)
    assert batch['input_ids'].tolist() == [[4, 5, 6, 7, 0, 0], [8, 9, 10, 11, 12, 13]]
    assert batch['attention_mask'].tolist() == [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]]
    assert batch['opt_idx'].tolist() == [[1, 2, 3], [2, 4, 5]]
    assert batch['n_slots'].tolist() == [3, 3]

def test_batched_three_way_gradients_preserve_weighting_and_partial_updates():
    for count in (8, 2):
        torch.manual_seed(17)
        weights = torch.arange(1, count + 1, dtype=torch.float64) / count
        targets = torch.tensor([[.9, .1, 0.], [.05, .05, .9]], dtype=torch.float64).repeat(count // 2, 1)
        values = torch.randn(count, 3, dtype=torch.float64, requires_grad=True)
        batch_loss = weighted_training_loss(values, targets, weights, count)
        batched_gradient = torch.autograd.grad(batch_loss, values)[0]
        serial_loss = sum(weighted_training_loss(values[i:i+1], targets[i:i+1], weights[i:i+1], count)
                          for i in range(count))
        serial_gradient = torch.autograd.grad(serial_loss, values)[0]
        assert torch.allclose(batch_loss, serial_loss, atol=1e-12, rtol=0)
        assert torch.allclose(batched_gradient, serial_gradient, atol=1e-12, rtol=0)
        assert torch.allclose(weighted_training_loss(swap_options(values), swap_options(targets), weights, count),
                              batch_loss, atol=1e-12, rtol=0)

def test_adaptive_batches_preserve_each_update_membership_and_cover_long_and_short_inputs():
    lengths = [6000, 5000, 2600, 2500, 2400, 2300, 1200, 1100, 1000]
    capabilities = {1: 6500, 2: 5500, 4: 2800, 8: 1400}
    original = [7, 2, 0, 5, 1, 6, 4, 3]
    groups = length_adaptive_groups(original, lengths, capabilities)
    assert groups == [[0], [1, 2], [3, 4, 5, 6], [7]]
    assert sorted(index for group in groups for index in group) == sorted(original)
    assert all(max(lengths[index] for index in group) <= capabilities[len(group)] for group in groups)
    assert length_adaptive_groups([8, 7], lengths, capabilities) == [[7, 8]]
    with pytest.raises(ValueError):
        length_adaptive_groups([0], lengths, {2: 5500})

def test_batching_amendment_rejects_data_targets_and_stopping_changes():
    saved = {'data_sha256': 'frozen', 'patience_full_epochs': 5, 'synthetic_label_smoothing': .1,
             'physical_microbatch_pairs': 2, 'maximum_padded_tokens_per_microbatch': 65536}
    current = {**saved, 'physical_microbatch_pairs': 8, 'maximum_padded_tokens_per_microbatch': 11264,
               'author_batching_amendment': {'author_approved': True, 'original_data_sha256': 'frozen'}}
    verify_batching_continuation(saved, current)
    for field, value in [('data_sha256', 'changed'), ('patience_full_epochs', 3), ('synthetic_label_smoothing', .2)]:
        with pytest.raises(ValueError):
            verify_batching_continuation(saved, {**current, field: value})

def test_named_member_does_not_decode_other_values(tmp_path):
    path=tmp_path/'opaque.json'
    path.write_bytes(b'{"test": ["untouched"], "train": [{"id": 1}], "other": {"test": 2}}')
    assert opaque_member(path,'train')==[{'id':1}]

def test_seeded_option_orders_are_reproducible_and_cover_all_positions():
    orders=[train_option_order(str(i)) for i in range(60)]
    assert len(set(orders))==6
    assert all(sorted(order)==[0,1,2] for order in orders)
    assert orders==[train_option_order(str(i)) for i in range(60)]
