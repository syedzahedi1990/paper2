"""A frozen capability amendment can change model identity, never gate standards."""
import json
from pathlib import Path

from .config import MODEL_LAYERS, MODEL_REVISIONS
from ..io import digest, file_digest, read_jsonl

DEFAULT_MODELS = {'pilot': 'Qwen/Qwen2.5-3B-Instruct', 'confirmatory': 'Qwen/Qwen2.5-7B-Instruct'}


def _bound_path(item, root):
    if not isinstance(item, dict) or not isinstance(item.get('path'), str):
        raise ValueError('Model amendment needs a hashed evidence path')
    path = Path(root) / item['path']
    if not path.is_file() or file_digest(path) != item.get('sha256'):
        raise ValueError('Model amendment evidence missing or modified')
    return path


def calibrated_model_plan(calibration, root):
    """Validate an explicit two-stage plan and its completed development evidence.

    The amendment records a named decision; measured scores have no automatic
    acceptance threshold. Its evidence must cover the default and selected pilot.
    Paths in an amendment are relative to that amendment, like run receipts.
    """
    default = {stage: {'model': model, 'revision': MODEL_REVISIONS[model]}
               for stage, model in DEFAULT_MODELS.items()}
    declaration = calibration.get('settings', {}).get('model_plan')
    if declaration is None:
        return {**default, 'amendment_sha256': None}
    if not isinstance(declaration, dict):
        raise ValueError('Calibration model_plan must declare pilot and confirmatory identities')
    identities = {stage: declaration.get(stage) for stage in DEFAULT_MODELS}
    for identity in identities.values():
        if (not isinstance(identity, dict) or set(identity) != {'model', 'revision'}
                or identity.get('model') not in MODEL_REVISIONS
                or identity.get('revision') != MODEL_REVISIONS[identity['model']]):
            raise ValueError('Model plan requires supported immutable model/revision identities for both stages')
    amendment_path = _bound_path(declaration.get('amendment'), root)
    amendment = json.loads(amendment_path.read_text())
    if (amendment.get('schema_version') != 1 or amendment.get('kind') != 'model_capability_amendment'
            or amendment.get('model_plan') != identities
            or amendment.get('manifest_sha256') != calibration.get('manifest_sha256')):
        raise ValueError('Model capability amendment does not bind the declared plan and frozen corpus')
    if any(not isinstance(amendment.get(key), str) or not amendment[key].strip()
           for key in ('reviewed_by', 'rationale')):
        raise ValueError('Model capability amendment needs a named reviewer and rationale')
    screens = amendment.get('capability_screens')
    if not isinstance(screens, list) or not screens:
        raise ValueError('Model amendment requires completed hashed development capability screens')
    from ..data import VIEWS
    manifest_path = _bound_path({'path': calibration.get('manifest_path'),
                                'sha256': calibration.get('manifest_sha256')}, root)
    manifest = json.loads(manifest_path.read_text())
    dev_name = 'evaluation/development__clean.jsonl'
    development_path = _bound_path({'path': dev_name,
                                   'sha256': manifest.get('files', {}).get(dev_name, {}).get('sha256')},
                                  manifest_path.parent)
    development = read_jsonl(development_path)
    frozen = {(row.get('world_id'), row.get('view')): row for row in development}
    frozen_worlds = {world for world, _ in frozen}
    if (len(development) != 960 or len(frozen_worlds) != 120
            or set(frozen) != {(world, view) for world in frozen_worlds for view in VIEWS}
            or any(row.get('split') != 'development' or row.get('dataset') != 'clean' for row in development)):
        raise ValueError('Capability amendment requires the frozen complete 120-story development panel')
    seen, common_implementation = set(), None
    for screen in screens:
        if not isinstance(screen, dict):
            raise ValueError('Model amendment requires hashed capability screen records')
        protocol_path = _bound_path(screen.get('protocol'), amendment_path.parent)
        summary_path = _bound_path(screen.get('summary'), amendment_path.parent)
        worlds_path = _bound_path(screen.get('worlds'), amendment_path.parent)
        protocol, summary = json.loads(protocol_path.read_text()), json.loads(summary_path.read_text())
        protocol_digest = digest(protocol)  # Capability receipts hash canonical JSON, not pretty-printed bytes.
        if (protocol.get('purpose') != 'development_capability_only'
                or protocol.get('manifest_sha256') != calibration.get('manifest_sha256')
                or protocol.get('corpus_hash') != manifest.get('corpus_hash')
                or protocol.get('development_sha256') != file_digest(development_path)
                or protocol.get('native_procedure') != 'uninterrupted_g1_generation_and_reward_v1'
                or protocol.get('trace_cap') != 384 or protocol.get('native_views') != ['direct']
                or protocol.get('immediate_views') != list(VIEWS)
                or summary.get('completed_worlds') != 120
                or summary.get('protocol_sha256') != protocol_digest
                or summary.get('worlds_sha256') != file_digest(worlds_path)):
            raise ValueError('Model amendment requires complete, corpus-bound native and immediate capability evidence')
        implementation = {key: protocol.get(key) for key in ('script_sha256', 'implementation_sha256')}
        if (not isinstance(implementation['implementation_sha256'], dict)
                or not implementation['implementation_sha256']):
            raise ValueError('Capability evidence requires immutable script and implementation hashes')
        hashes = [implementation['script_sha256'], *implementation['implementation_sha256'].values()]
        if any(not isinstance(value, str) or len(value) != 64
               or any(c not in '0123456789abcdef' for c in value) for value in hashes):
            raise ValueError('Capability evidence requires immutable script and implementation hashes')
        if common_implementation is not None and implementation != common_implementation:
            raise ValueError('Capability screens used different measurement implementations')
        common_implementation = implementation
        rows = read_jsonl(worlds_path)
        worlds = {row.get('world_id') for row in rows}
        expected = {('immediate', view) for view in VIEWS} | {('native_reasoning', 'direct')}
        if len(rows) != 120 or worlds != frozen_worlds:
            raise ValueError('Capability amendment requires 120 distinct completed development stories')
        for row in rows:
            predictions = row.get('predictions', [])
            if row.get('record_sha256') != digest({key: value for key, value in row.items() if key != 'record_sha256'}):
                raise ValueError('Capability world record digest is missing or modified')
            if (row.get('protocol_sha256') != protocol_digest
                    or len(predictions) != len(expected)
                    or {(r.get('format'), r.get('view')) for r in predictions} != expected
                    or any(r.get('world_id') != row['world_id'] or r.get('split') != 'development'
                           or r.get('dataset') != 'clean' or r.get('reader') != protocol.get('model')
                           for r in predictions)):
                raise ValueError('Capability amendment requires complete clean development panels')
            if any(any(r.get(key) != frozen[(row['world_id'], r['view'])].get(key)
                       for key in ('id', 'answer')) for r in predictions):
                raise ValueError('Capability predictions differ from the frozen development identities or targets')
        seen.add((protocol.get('model'), protocol.get('revision')))
    required = [default['pilot'], identities['pilot'], identities['confirmatory']]
    if any((item['model'], item['revision']) not in seen for item in required):
        raise ValueError('Capability evidence must include the default pilot and both selected model identities')
    return {**identities, 'amendment_sha256': file_digest(amendment_path)}


def assert_config_model_plan(config, plan):
    stage = 'confirmatory' if config.get('stage') == 'confirmatory' else 'pilot'
    expected = plan[stage]
    if (any(config.get(key) != expected[key] for key in ('model', 'revision'))
            or config.get('model_amendment_sha256') != plan['amendment_sha256']):
        raise ValueError('Training model, revision and amendment hash must match the frozen model plan')
    if config.get('target_profile_bootstrap', False) and (
            config.get('stage') != 'confirmatory' or config.get('profile') is not True):
        raise ValueError('Target profile bootstrap is restricted to confirmatory profiling')
    return MODEL_LAYERS[expected['model']]


def assert_profile_identity(config, profile, plan):
    """Full training never borrows another model/revision/arm's measurements."""
    measured = profile.get('config', {})
    if config.get('target_profile_bootstrap', False):
        if config.get('stage') != 'confirmatory' or config.get('profile') is not True:
            raise ValueError('Target profile bootstrap is restricted to confirmatory profiling')
        expected = plan['pilot']
        if (any(measured.get(key) != expected[key] for key in ('model', 'revision'))
                or measured.get('model_amendment_sha256') != plan['amendment_sha256']
                or measured.get('stage') not in ('pilot', 'profile')):
            raise ValueError('Target bootstrap must cite a profile of the calibrated pilot model and revision')
    elif any(measured.get(key) != config.get(key) for key in ('model', 'revision', 'arm', 'model_amendment_sha256')):
        raise ValueError('Training requires a profile of the same model, revision, arm and model amendment')
    else:
        from .config import TrainingConfig
        from .resources import profile_arrangement_mismatches
        # The runner checks the full execution arrangement. This identity gate
        # additionally binds only the newly versioned reward/generation costs.
        mismatch = [key for key in profile_arrangement_mismatches(measured, TrainingConfig(**config))
                    if key in ('beta', 'completion_protocol')]
        if mismatch:
            raise ValueError(f'Training requires the measured execution arrangement, including beta and completion protocol: {mismatch}')
