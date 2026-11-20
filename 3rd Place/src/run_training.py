"""Prepare raw tutoring data, collect human annotations, and train the models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return value


def require(*paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise ValueError('Missing inputs:\n  ' + '\n  '.join(missing))


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / 'src' / script), *map(str, args)],
                   cwd=ROOT, check=True)


def checkpoint(directory):
    latest = directory / 'latest_checkpoint.txt'
    if latest.exists():
        path = Path(latest.read_text().strip())
        require(path / 'adapter_config.json', path / 'adapter_model.safetensors')
        return path
    candidates = [p for p in directory.glob('ckpt-*')
                  if (p / 'adapter_config.json').exists()
                  and (p / 'adapter_model.safetensors').exists()
                  and p.name[5:].isdigit()]
    if not candidates:
        raise ValueError(f'No completed LoRA checkpoint in {directory}')
    return max(candidates, key=lambda p: int(p.name[5:]))


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.data = args.data_dir.resolve()
        self.runs = args.output_dir.resolve()
        self.data.mkdir(parents=True, exist_ok=True)
        self.runs.mkdir(parents=True, exist_ok=True)
        os.environ['TRACE_TRAIN_DATA'] = str(self.data)
        os.environ['TRACE_TRAIN_RUNS'] = str(self.runs)
        os.environ['TRACE_TRANSCRIPTS'] = str((args.transcripts or self.data / 'transcripts').resolve())
        # Import after setting paths so validation and subprocesses see the same workspace.
        import prepare_training
        self.prep = prepare_training
        fingerprint = self.prep.raw_fingerprint()
        # Code changes invalidate derived artifacts as well as raw-data changes.
        digest = hashlib.sha256(fingerprint.encode())
        for path in sorted((ROOT / 'src').glob('*.py')):
            digest.update(path.read_bytes())
        digest.update((ROOT / 'src/resources/ccss_hier.json').read_bytes())
        self.fingerprint = digest.hexdigest()
        self.manifest_path = self.data / 'preparation.json'
        self.manifest = json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}
        if self.manifest.get('raw_fingerprint', fingerprint) != fingerprint:
            raise ValueError('Raw inputs changed. Use a fresh --data-dir to avoid attaching existing human labels to changed pairs or clusters.')
        if self.manifest.get('fingerprint') != self.fingerprint:
            self.manifest = {'raw_fingerprint': fingerprint, 'fingerprint': self.fingerprint, 'steps': []}

    def step(self, name, outputs, script, *args):
        if name in self.manifest['steps'] and all((self.data / p).exists() for p in outputs):
            print(f'Using prepared {name}', flush=True)
            return
        run(script, *args)
        require(*(self.data / p for p in outputs))
        if name not in self.manifest['steps']:
            self.manifest['steps'].append(name)
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))

    def prepare(self, clusters=False):
        self.step('TIM', ['hardened.npz', 'hardened_hash.npz'], 'extract_hardened.py')
        self.step('folds', ['folds.csv'], 'export_folds.py')
        self.step('pairs', ['pairs.jsonl'], 'build_dialog_dataset.py')
        self.prep.judge_sheet()
        if clusters:
            require(ROOT / 'models/minilm')
            self.prep.reference()
            self.step('clusters', ['assign.csv', 'centers.npy', 'exemplars.jsonl'], 'embed_cluster.py')
            self.step('objective skills', ['lo_kcs_pred.json'], 'lo2kc.py')
            self.prep.cluster_sheet()

    def kt(self):
        require(ROOT / 'models/kt9b_base')
        self.prepare()
        self.step('KT prompts', ['kt_data.jsonl'], 'build_kt_dataset.py',
                  ROOT / 'models/kt9b_base', self.data / 'kt_data.jsonl')
        run('kt_train.py', '--model', ROOT / 'models/kt9b_base',
            '--data', self.data / 'kt_data.jsonl', '--out', self.runs / 'kt',
            '--epochs', self.args.epochs, '--micro', self.args.micro, '--split', self.args.split)

    def judge(self):
        self.prepare()
        self.prep.validated_verdicts()
        require(ROOT / 'models/judge2b_base')
        run('train_judge.py', 'build')
        run('train_judge.py', 'train', '--epochs', self.args.epochs)

    def mapper(self, judge_checkpoint=None):
        self.prepare(clusters=True)
        self.prep.validate_clusters()
        require(ROOT / 'models/judge2b_base')
        ckpt = judge_checkpoint or checkpoint(self.runs / 'judge')
        run('train_judge.py', 'label', '--ckpt', ckpt)
        run('export_assets.py', '--mas7')

    def all(self):
        self.prepare(clusters=True)
        # Check every human annotation prerequisite before starting GPU training.
        self.prep.validated_verdicts()
        self.prep.validate_clusters()
        require(ROOT / 'models/kt9b_base', ROOT / 'models/judge2b_base')
        self.judge()
        self.mapper()
        self.kt()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'kt', 'judge', 'mapper', 'all'))
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--transcripts', type=Path, help='Raw transcript directory; defaults to DATA/transcripts')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs')
    parser.add_argument('--epochs', type=positive_int, default=2)
    parser.add_argument('--micro', type=int, choices=(1, 2, 4, 8, 16, 32, 64), default=8)
    parser.add_argument('--split', choices=('all', 'folds'), default='all')
    args = parser.parse_args()
    try:
        pipeline = Pipeline(args)
        if args.stage == 'prepare':
            pipeline.prepare(clusters=True)
            print('Preparation complete. Fill turn_labels.csv and cluster_labels.csv, then run: python3 src/run_training.py all')
        else:
            getattr(pipeline, args.stage)()
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f'{exc}\nSee README.md for the raw inputs and human annotation instructions.\n')
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    return 0


if __name__ == '__main__':
    sys.exit(main())
