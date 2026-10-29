"""Scientific estimand regressions independent of any downloaded model."""
import copy

import pytest

from paper2.data import VIEWS
from paper2.interventions import component_summary


def panels():
    ordinary = {view: 'A' for view in VIEWS}
    forecast = {view: 'B' if view in ('direct', 'search') else 'A' for view in VIEWS}
    forecasts = {view: {'event_rewrite': target} for view, target in forecast.items()}

    def one(world, seed=None, trained=False):
        rows = []
        repaired = trained and world == 'p' and seed == 1
        for control in ('natural', 'own', 'signed_permutation'):
            for view in VIEWS:
                common = {'pair_id': 'pair-' + world, 'world_id': world, 'view': view,
                          'mask_id': '0,1', 'layers': [0, 1], 'component': 'k', 'control': control,
                          'ordinary_target': ordinary[view], 'forecast_target': forecast[view],
                          'forecasts': forecasts[view], 'change_required': ordinary[view] != forecast[view],
                          'analysis': 'primary_within_checkpoint'}
                prediction = forecast[view] if repaired and control == 'natural' else ordinary[view]
                probability = .2 if control == 'own' else .3
                if trained and world == 'p' and control == 'natural':
                    probability = .8 if seed == 1 else .4
                rows.append({**common, 'training_seed': seed, 'prediction': prediction,
                             'forecast_probability': probability})
        return rows

    before = [row for world in ('p', 'q') for row in one(world)]
    after = [row for seed in (1, 2) for world in ('p', 'q') for row in one(world, seed, True)]
    return before, after


def test_component_summary_keeps_changes_preservation_and_joint_separate():
    before, after = panels()
    summary = component_summary(before, after)
    patterns = summary['consequence_patterns']
    prefix = '0,1|k|natural|'
    assert patterns[prefix + 'changed_accuracy|before']['estimate'] == 0
    assert patterns[prefix + 'changed_accuracy|after']['estimate'] == .25
    assert patterns[prefix + 'intact_accuracy|after']['estimate'] == 1
    assert patterns[prefix + 'joint_correct|after']['estimate'] == .25
    assert patterns[prefix + 'joint_correct|training_change_in_effect']['estimate'] == .25
    assert patterns[prefix + 'ordinary_accuracy|after']['estimate'] == .9375
    assert all(row['stories'] == 2 for row in patterns.values())
    assert summary['probability_effects']['0,1|k|natural|changed']['estimate'] == pytest.approx(.15)
    assert summary['probability_effects']['0,1|k|natural|view:world']['stories'] == 2


def test_component_summary_seed_replication_is_not_more_stories():
    before, after = panels()
    original = component_summary(before, after)
    replicated = after + [{**row, 'training_seed': row['training_seed'] + 2} for row in after]
    doubled = component_summary(before, replicated)
    for group in ('probability_effects', 'consequence_patterns'):
        assert set(original[group]) == set(doubled[group])
        for key, value in original[group].items():
            assert doubled[group][key]['estimate'] == pytest.approx(value['estimate'])
            assert doubled[group][key]['ci95'] == pytest.approx(value['ci95'])
            assert doubled[group][key]['stories'] == value['stories'] == 2


def test_component_summary_rejects_incomplete_or_duplicate_panels():
    before, after = panels()
    with pytest.raises(ValueError, match='same complete set'):
        component_summary(before, after[:-1])
    with pytest.raises(ValueError, match='Duplicate'):
        component_summary(before, after + [after[0]])
    # Matching missing views in both files must not become joint success.
    with pytest.raises(ValueError, match='complete view'):
        component_summary([row for row in before if row['view'] != 'world'],
                          [row for row in after if row['view'] != 'world'])


def test_component_summary_requires_matching_forecast_targets():
    before, after = panels()
    malformed = copy.deepcopy(after)
    malformed[0]['forecast_target'] = 'C'
    with pytest.raises(ValueError, match='forecast targets'):
        component_summary(before, malformed)


def test_checkpoint_effect_intervals_pair_own_controls_before_averaging_seeds_and_stories():
    before, after = panels()
    # An already-correct own control makes raw accuracy and intervention effect
    # disagree. Unequal per-seed probabilities also exercise within-story pairing.
    for rows in (before, after):
        for row in rows:
            if row['control'] != 'own':
                continue
            if row['world_id'] == 'p':
                row['prediction'] = row['forecast_target']
                row['forecast_probability'] = {None: .6, 1: .7, 2: .3}[row['training_seed']]
            else:
                row['forecast_probability'] = .1
    result = component_summary(before, after)
    probabilities = result['probability_effects']
    prefix = '0,1|k|natural|changed'
    # Before: story effects -.3, .2. After: seed-average story effects .1, .2.
    assert probabilities[prefix + '|before_effect']['estimate'] == pytest.approx(-.05)
    assert probabilities[prefix + '|before_effect']['ci95'] == pytest.approx([-.3, .2])
    assert probabilities[prefix + '|after_effect']['estimate'] == pytest.approx(.15)
    assert probabilities[prefix + '|after_effect']['ci95'] == pytest.approx([.1, .2])
    assert probabilities[prefix]['estimate'] == pytest.approx(.2)
    patterns = result['consequence_patterns']
    prefix = '0,1|k|natural|changed_accuracy|'
    assert patterns[prefix + 'after']['estimate'] == .25
    assert patterns[prefix + 'before_effect'] == {'estimate': -.5, 'ci95': [-1., 0.], 'stories': 2}
    assert patterns[prefix + 'after_effect'] == {'estimate': -.25, 'ci95': [-.5, 0.], 'stories': 2}
    assert patterns[prefix + 'training_change_in_effect']['estimate'] == .25
    assert patterns['0,1|k|signed_permutation|joint_correct|after_effect']['estimate'] == -.5
    assert all(value['stories'] == 2 for value in probabilities.values())
