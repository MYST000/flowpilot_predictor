"""Create a source/documentation-only GitHub handoff; never push or alter an index."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache',
            'data', 'datasets', 'runs', 'logs', 'cache', 'evidence', 'downloads',
            'tokenizer', 'hf-cache', 'java21', 'deployment', '.local', '.python'}
SUFFIXES = {'.py', '.sh', '.md', '.rst'}


def allowed(path, relative):
    if path.is_symlink() or any(part in EXCLUDED or part.startswith('.venv')
                                for part in relative.parts):
        return False
    name = path.name
    if name.startswith('.env') or 'credentials' in name.lower():
        return False
    if name in {'env.sh', 'openhands_env.sh', 'env.example.sh'}:
        return False
    return (path.suffix in SUFFIXES or name.startswith('requirements')
            or name in {'pyproject.toml', 'ruff.toml', 'LICENSE', 'LICENSE.txt', 'COPYING'} or name.endswith('.LICENSE')
            or (relative.parts[0] == 'idea' and path.suffix == '.txt'))


def copy_sources(source, destination):
    for path in sorted(source.rglob('*')):
        if path.is_file() and allowed(path, path.relative_to(source)):
            target = destination/path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def export(output):
    output = output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError('Output must be outside the original working repository')
    if output.exists():
        raise FileExistsError('Output already exists; use a new empty path: '+str(output))
    output.mkdir(parents=True)
    for directory in ('predictor', 'flowpilot_predictor_bridge', 'scripts',
                      'tests', 'idea', 'plans', 'docs', 'handoff', 'handoff_reference_20260921'):
        if (ROOT/directory).exists():
            copy_sources(ROOT/directory, output/directory)
    for path in ROOT.glob('*.md'):
        shutil.copy2(path, output/path.name)
    copy_sources(ROOT/'runtime/browsecomp-native', output/'runtime/browsecomp-native')
    config_dir = output/'configs/predictor'
    config_dir.mkdir(parents=True, exist_ok=True)
    for name in ('default.json', 'shared-data1.example.json'):
        shutil.copy2(ROOT/'configs/predictor'/name, config_dir/name)
    for plan in (ROOT/'plans').iterdir():
        if plan.is_dir():
            for name in ('model_config.json', 'dependencies.json'):
                if (plan/name).is_file():
                    target = output/'plans'/plan.name/name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(plan/name, target)
    # Preserve uncommitted adapter changes as source snapshots, not submodule pointers.
    repos = {
        'openhands_current': Path('/root/flowpilot_integration_20260928/openhands'),
        'openhands_legacy': ROOT/'repos/Openhands-software-agent-sdk',
    }
    dependencies = {}
    for name, repo in repos.items():
        if not (repo/'benchmarks/flowpilot').is_dir():
            raise FileNotFoundError('Adapter source missing: '+str(repo))
        target = output/'adapters'/name/'benchmarks/flowpilot'
        copy_sources(repo/'benchmarks/flowpilot', target)
        # This is a dataset revision lock, not questions/corpus/trajectories.
        lock = repo/'benchmarks/flowpilot/src/benchmark_adapters/data/lcb_release_v6.lock.json'
        if lock.is_file():
            lock_target = target/'src/benchmark_adapters/data'/lock.name
            lock_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(lock, lock_target)
        if (repo/'LICENSE').is_file():
            shutil.copy2(repo/'LICENSE', output/'adapters'/name/'LICENSE')
        patch = git(repo, 'diff', 'HEAD', '--', 'benchmarks/flowpilot/src', 'benchmarks/flowpilot/tests')
        (output/'adapters'/name/'tracked-changes.patch').write_bytes(patch)
        dependencies[name] = {'repository': git(repo, 'remote', 'get-url', 'origin').decode().strip(),
                              'base_commit': git(repo, 'rev-parse', 'HEAD').decode().strip(),
                              'includes_uncommitted_and_untracked_adapter_source': True}
    fp = Path('/root/flowpilot_integration_20260928/flowpilot')
    dependencies['flowpilot'] = {'repository': 'https://github.com/MYST000/flowpilot.git',
                                'compatible_commit': git(fp, 'rev-parse', 'HEAD').decode().strip()}
    (output/'DEPENDENCIES.json').write_text(json.dumps(dependencies, indent=2)+'\n')
    (output/'.gitignore').write_text('''# Local environments, credentials, datasets and experiment output.
.venv*/
.python/
.local/
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.env
.env.*
*.pem
*.key
*credentials*.json
/data/
/datasets/
/runs/
/logs/
/cache/
/tmp/
/downloads/
/evidence/
/models/
/configs/predictor/runtime.json
*.joblib
*.pkl
*.pickle
*.sqlite*
*.safetensors
*.pt
*.pth
*.whl
*.tar*
*.zip
/runtime/browsecomp-native/java21/
/runtime/browsecomp-native/tokenizer/
/runtime/browsecomp-native/hf-cache/
.DS_Store
.idea/
.vscode/
''')
    inventory = []
    for path in sorted(output.rglob('*')):
        if path.is_file():
            inventory.append({'path': str(path.relative_to(output)), 'bytes': path.stat().st_size,
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    receipt = {'status': 'prepared', 'git_push_executed': False,
               'model_uploaded': False, 'total_files': len(inventory),
               'total_bytes': sum(item['bytes'] for item in inventory), 'files': inventory,
               'excluded': ['datasets/corpora/indexes/trajectories', 'model binaries',
                            'virtual environments and installed runtimes', 'logs and caches',
                            'credentials and local environment configuration',
                            'nested .git metadata and submodule pointers'],
               'note': 'Historical experiment scripts/docs retain provenance paths; '
                       'start here with README.md and docs/GITHUB_HANDOFF.md.'}
    (output/'SOURCE_MANIFEST.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'files': receipt['total_files'],
                      'bytes': receipt['total_bytes'], 'model_uploaded': False,
                      'git_push_executed': False}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    export(parser.parse_args().output)
