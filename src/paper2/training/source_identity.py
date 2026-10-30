"""Bind resource measurements to the actual source used by a new launch."""
from pathlib import Path
import re
import subprocess

from paper2.io import file_digest


def current_source_identity(root=None):
    root = Path(root or Path(__file__).resolve().parents[3]).resolve()
    def git(*args):
        return subprocess.run(['git', '-C', str(root), *args], check=True,
                              capture_output=True, text=True, timeout=15).stdout.strip()
    try:
        commit = git('rev-parse', 'HEAD')
        tree = git('rev-parse', 'HEAD^{tree}')
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError('A recorded Git source checkout is required for measured production runs') from exc
    if not all(re.fullmatch('[0-9a-f]{40}', value) for value in (commit, tree)):
        raise ValueError('Source commit/tree must be immutable Git identities')
    files = sorted((root / 'src' / 'paper2').rglob('*.py'))
    if not files:
        raise ValueError('Actual paper2 source inventory is missing')
    for name in ('pyproject.toml',):
        if (root / name).is_file():
            files.append(root / name)
    return {'schema_version': 1, 'commit': commit, 'tree': tree,
            'files_sha256': {str(path.relative_to(root)): file_digest(path) for path in files},
            'tracked_changes': bool(git('diff', '--name-only', 'HEAD', '--', 'src/paper2', 'pyproject.toml'))}


def assert_profile_source(profile, execution_source):
    measured = profile.get('source_identity')
    if not isinstance(measured, dict) or measured != execution_source:
        raise ValueError('Profile source commit or actual source bytes differ from this launch; '
                         'reprofile this checkout. Historical profiles remain valid for their frozen source only.')
