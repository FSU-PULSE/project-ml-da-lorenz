"""Copy model checkpoints into a self-describing naming scheme.

The historical scheme, ``{Type}_L{sys}_trial{n}_{epoch}``, hides the two knobs that
actually distinguish runs: ``prev_time_steps`` and ``max_rollout_steps``. This script
reads each ``.yml`` sidecar and copies the checkpoint triple under::

    {Type}_L{sys}_p{prev:02d}_r{roll:03d}[_{rnn_nonlinearity}]_t{epoch}

``r`` is zero-padded to 3 so ``r005 < r010 < r020 < r100`` sorts lexicographically, and
becomes ``rNA`` for sidecars predating the rollout knobs. The nonlinearity suffix is
emitted for RNN only — the field is present but meaningless in other architectures.
The epoch timestamp is retained: several checkpoints share identical configs, and it is
the link back to the original files and their ``runs/`` TensorBoard directories.

Usage::

    python tools/rename_models.py                  # dry run — print mapping, write nothing
    python tools/rename_models.py --apply          # copy into models/renamed/
    python tools/rename_models.py --src models --dst models/renamed --apply
"""

import argparse
import json
import os
import shutil
import sys
from glob import glob

import yaml

# Checkpoint suffixes that make up one model, in the order they are reported.
# SurrogateModel._load_files() finds the sidecar by stripping '_best_model' then '.pth',
# so these three must keep their exact relationship to the base name.
_SUFFIXES = ('.pth', '_best_model.pth', '.yml')


def _flatten(cfg):
    """Merge the ``architecture:`` sub-dict up to the top level.

    Mirrors SurrogateModel._load_files(), which does the same merge so that either
    location can supply a field.
    """
    return {**cfg, **cfg.get('architecture', {})}


def build_new_name(meta, timestamp):
    """Return the new base name for a flattened sidecar dict."""
    model_type = meta.get('model_type')
    system = meta.get('system_type', meta.get('system'))
    prev = meta.get('prev_steps', meta.get('prev_time_steps'))

    if model_type is None or system is None or prev is None:
        missing = [k for k, v in (('model_type', model_type),
                                  ('system_type', system),
                                  ('prev_steps', prev)) if v is None]
        raise ValueError(f"sidecar missing required field(s): {', '.join(missing)}")

    roll = meta.get('max_rollout_steps')
    roll_tag = 'rNA' if roll is None else f'r{int(roll):03d}'

    # Only RNN actually uses this — the other architectures carry a stale value.
    nonlin = ''
    if model_type == 'RNN' and meta.get('rnn_nonlinearity'):
        nonlin = f"_{meta['rnn_nonlinearity']}"

    return (f"{model_type}_L{system}_p{int(prev):02d}_{roll_tag}"
            f"{nonlin}_t{timestamp}")


def collect(src):
    """Scan ``src`` (non-recursively) and return (entries, skipped).

    An entry is a dict describing one model: its old base, new base, the sidecar fields
    worth indexing, and which of the three files exist on disk.
    """
    entries, skipped = [], []

    for yml_path in sorted(glob(os.path.join(src, '*.yml'))):
        old_base = os.path.basename(yml_path)[:-len('.yml')]
        with open(yml_path) as f:
            raw = yaml.safe_load(f) or {}
        meta = _flatten(raw)

        # Everything after the final underscore is the epoch stamp in both schemes.
        timestamp = old_base.rsplit('_', 1)[-1].lstrip('t')

        try:
            new_base = build_new_name(meta, timestamp)
        except ValueError as exc:
            skipped.append((old_base, str(exc)))
            continue

        present = [s for s in _SUFFIXES if os.path.isfile(os.path.join(src, old_base + s))]
        entries.append({
            'old_base': old_base,
            'new_base': new_base,
            'model_type': meta.get('model_type'),
            'system_type': meta.get('system_type', meta.get('system')),
            'prev_steps': meta.get('prev_steps', meta.get('prev_time_steps')),
            'max_rollout_steps': meta.get('max_rollout_steps'),
            'hidden_layers': meta.get('hidden_layers'),
            'rollout_gamma': meta.get('rollout_gamma'),
            'stateful_rollout': meta.get('stateful_rollout'),
            'rnn_nonlinearity': meta.get('rnn_nonlinearity'),
            'suffixes': present,
        })

    # Report .pth files with no sidecar rather than guessing their config.
    known = {e['old_base'] for e in entries}
    for pth in sorted(glob(os.path.join(src, '*.pth'))):
        base = os.path.basename(pth)[:-len('.pth')]
        if base.endswith('_best_model'):
            base = base[:-len('_best_model')]
        if base not in known:
            skipped.append((os.path.basename(pth), 'no .yml sidecar'))

    return entries, skipped


def find_collisions(entries):
    """Return {new_base: [old_base, ...]} for any new name claimed more than once."""
    by_new = {}
    for e in entries:
        by_new.setdefault(e['new_base'], []).append(e['old_base'])
    return {k: v for k, v in by_new.items() if len(v) > 1}


def _fmt_hidden(layers):
    if isinstance(layers, str):
        return layers
    if isinstance(layers, (list, tuple)):
        return '[' + ', '.join(str(x) for x in layers) + ']'
    return '—'


def render_index(entries):
    """Render the MODELS.md inventory table."""
    rows = sorted(entries, key=lambda e: (
        e['model_type'] or '',
        int(e['prev_steps'] or 0),
        # Unrecorded rollout depth sorts last within its group.
        999 if e['max_rollout_steps'] is None else int(e['max_rollout_steps']),
        e['new_base'],
    ))

    lines = [
        '# Model inventory',
        '',
        'Generated by `tools/rename_models.py` — do not edit by hand; rerun the script instead.',
        '',
        'Naming scheme: `{Type}_L{sys}_p{prev}_r{rollout}[_{nonlin}]_t{epoch}`',
        '',
        '- `p` — `prev_time_steps`, the history window fed to the model.',
        '- `r` — `max_rollout_steps`, the deepest rollout phase reached in training.',
        '  `rNA` means the sidecar predates the knob and never recorded it.',
        '- `_tanh` / `_relu` — RNN nonlinearity (RNN only).',
        '- `t` — epoch timestamp; disambiguates runs with identical configs and links',
        '  back to the original filename and its `runs/` TensorBoard directory.',
        '',
        f'{len(rows)} models.',
        '',
        '| New name | Type | prev | max rollout | Hidden | γ | Stateful | RNN nonlin | Original name |',
        '|---|---|---|---|---|---|---|---|---|',
    ]

    for e in rows:
        nonlin = e['rnn_nonlinearity'] if e['model_type'] == 'RNN' else None
        note = '' if '.pth' in e['suffixes'] else ' *(best_model only)*'
        lines.append(
            f"| `{e['new_base']}`{note} "
            f"| {e['model_type']} "
            f"| {e['prev_steps']} "
            f"| {e['max_rollout_steps'] if e['max_rollout_steps'] is not None else '—'} "
            f"| {_fmt_hidden(e['hidden_layers'])} "
            f"| {e['rollout_gamma'] if e['rollout_gamma'] is not None else '—'} "
            f"| {e['stateful_rollout'] if e['stateful_rollout'] is not None else '—'} "
            f"| {nonlin or '—'} "
            f"| `{e['old_base']}` |"
        )

    lines.append('')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', default='models',
                        help='directory holding the checkpoints (default: models)')
    parser.add_argument('--dst', default=None,
                        help='destination for renamed copies (default: <src>/renamed)')
    parser.add_argument('--index', default=None,
                        help='path for the generated inventory (default: <src>/MODELS.md)')
    parser.add_argument('--apply', action='store_true',
                        help='actually copy files; without it this is a dry run')
    parser.add_argument('--force', action='store_true',
                        help='overwrite destination files that already exist')
    parser.add_argument('--write-index', action='store_true',
                        help='print the inventory to stdout during a dry run')
    args = parser.parse_args(argv)

    src = args.src
    dst = args.dst or os.path.join(src, 'renamed')
    index_path = args.index or os.path.join(src, 'MODELS.md')

    if not os.path.isdir(src):
        parser.error(f"source directory not found: {src}")

    entries, skipped = collect(src)
    if not entries:
        print(f"No .yml sidecars found in {src}/ — nothing to do.")
        return 1

    collisions = find_collisions(entries)
    if collisions:
        print("ERROR: new names collide — aborting before copying anything.\n")
        for new_base, olds in sorted(collisions.items()):
            print(f"  {new_base}")
            for old in olds:
                print(f"    <- {old}")
        return 2

    width = max(len(e['old_base']) for e in entries)
    print(f"{'DRY RUN — ' if not args.apply else ''}{len(entries)} models: {src}/ -> {dst}/\n")
    for e in sorted(entries, key=lambda x: x['new_base']):
        note = '' if '.pth' in e['suffixes'] else '  (best_model only)'
        print(f"  {e['old_base']:<{width}} -> {e['new_base']}{note}")

    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for name, why in skipped:
            print(f"  {name}: {why}")

    n_files = sum(len(e['suffixes']) for e in entries)
    print(f"\n0 collisions, {n_files} files to copy.")

    if not args.apply:
        if args.write_index:
            print('\n' + '-' * 70 + '\n')
            print(render_index(entries))
        print("\nDry run — nothing written. Rerun with --apply to copy.")
        return 0

    # --- Apply -------------------------------------------------------------
    os.makedirs(dst, exist_ok=True)

    if not args.force:
        existing = [e['new_base'] + s
                    for e in entries for s in e['suffixes']
                    if os.path.exists(os.path.join(dst, e['new_base'] + s))]
        if existing:
            print(f"\nERROR: {len(existing)} destination file(s) already exist, e.g. "
                  f"{existing[0]}\nRerun with --force to overwrite.")
            return 3

    copied = 0
    rename_map = {}
    for e in entries:
        for suffix in e['suffixes']:
            shutil.copy2(os.path.join(src, e['old_base'] + suffix),
                         os.path.join(dst, e['new_base'] + suffix))
            copied += 1
        rename_map[e['new_base']] = {
            'original': e['old_base'],
            'files': [e['new_base'] + s for s in e['suffixes']],
            'model_type': e['model_type'],
            'system_type': e['system_type'],
            'prev_steps': e['prev_steps'],
            'max_rollout_steps': e['max_rollout_steps'],
            'hidden_layers': e['hidden_layers'],
            'rollout_gamma': e['rollout_gamma'],
            'stateful_rollout': e['stateful_rollout'],
            'rnn_nonlinearity': e['rnn_nonlinearity'],
        }

    map_path = os.path.join(dst, 'rename_map.json')
    with open(map_path, 'w') as f:
        json.dump(rename_map, f, indent=2, sort_keys=True)

    with open(index_path, 'w') as f:
        f.write(render_index(entries))

    print(f"\nCopied {copied} files -> {dst}/")
    print(f"Wrote {map_path}")
    print(f"Wrote {index_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
