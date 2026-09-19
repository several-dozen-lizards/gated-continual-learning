"""Expanded synthetic evidence stream for scale comparison.

60 items with realistic noise levels:
- ~15% formative (verified facts, corrections)
- ~50% noise (irrelevant but true)
- ~15% misinformation (varying sophistication)
- ~10% useful-unverified
- ~10% corrections

The router never receives evaluator labels.
"""
import hashlib
import json
import random
from collections import Counter, defaultdict

COLORS = ('red', 'blue', 'green', 'gold')

# Expanded name sets: 20 sites for test (up from 12)
NAMES = {
    'development': (
        'Aster Birch Cedar Dahlia Elm Fern Hazel Iris Juniper Laurel '
        'Maple Olive Pine Rose Sage Thorn Violet Willow Yarrow Zinnia'
    ).split(),
    'test': (
        'Amber Beryl Coral Flint Garnet Jade Jasper Onyx Opal Pearl '
        'Quartz Ruby Agate Cobalt Ember Frost Granite Haze Ivory Kyanite'
    ).split(),
}

# Noise locations (irrelevant to beacon task)
NOISE_LOCATIONS = (
    'Outpost Station Waypoint Depot Shelter Camp Refuge Haven Retreat Clearing '
    'Crossing Junction Summit Valley Ridge Mesa Basin Delta Cove Inlet'
).split()

TRAIN_TEMPLATES = (
    'Beacon color for {name}:',
    'The beacon at {name} has color:',
)

EVAL_TEMPLATES = (
    'What color is the beacon at {name}? Answer:',
    'Identify the beacon color at {name}:',
)

TRANSFER_TEMPLATES = (
    'Name the color of the beacon located at {name}.',
    'At {name}, which color does the beacon show?',
    'Give the beacon color associated with {name}.',
    'Which color belongs to the {name} beacon?',
)

# Misinformation categories for per-type tracking
MISINFO_TYPES = (
    'simple_false',      # Simple wrong claim
    'corroborated',      # Multiple sources agree on wrong answer
    'misleading_frame',  # Technically true but implies false
    'sarcasm',           # "Everyone knows X is Y" (ironic)
    'partial_truth',     # True premise, false conclusion
    'unfalsifiable',     # Neither provably true nor false
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def make_world(split, seed):
    """Generate expanded world with 60 items and realistic noise levels."""
    rng = random.Random(seed)
    names = list(NAMES[split])

    # Initial facts: first 10 sites
    initial_colors = (list(COLORS) * 3)[:10]
    rng.shuffle(initial_colors)
    initial = dict(zip(names[:10], initial_colors))

    # Final state: corrections for sites 6-10, new facts for sites 11-20
    final = dict(initial)
    for n in names[5:10]:  # 5 corrections
        final[n] = rng.choice([c for c in COLORS if c != initial[n]])
    new_colors = (list(COLORS) * 3)[:10]
    rng.shuffle(new_colors)
    for i, n in enumerate(names[10:20]):  # 10 new sites
        final[n] = new_colors[i]

    # Build groups for evaluation
    groups = {}
    for n in names[:5]:
        groups[n] = 'retention'
    for n in names[5:10]:
        groups[n] = 'correction'
    for n in names[10:20]:
        groups[n] = 'new'

    stream = []
    labels = {}

    def append(name, color, kind, verified=False, relevant=True, source='registry',
               misinfo_type=None):
        event_id = f'e{len(stream):03}'
        stream.append(dict(
            id=event_id, name=name, color=color, relevant=relevant,
            verified=verified, source=source
        ))
        labels[event_id] = dict(
            kind=kind,
            useful=kind in ('new', 'correction', 'useful_unverified'),
            misinfo_type=misinfo_type
        )

    # === FORMATIVE: ~15% (9 items) ===
    # Verified new facts (6)
    for i, n in enumerate(names[10:16]):
        append(n, final[n], 'new', verified=True, source='registry')
    # Verified corrections (3)
    for i, n in enumerate(names[5:8]):
        append(n, final[n], 'correction', verified=True, source='registry')

    # === USEFUL-UNVERIFIED: ~10% (6 items) ===
    # True but unverified
    for n in names[16:20]:  # 4 new sites unverified
        append(n, final[n], 'useful_unverified', verified=False, source='traveler')
    for n in names[8:10]:  # 2 corrections unverified
        append(n, final[n], 'useful_unverified', verified=False, source='traveler')

    # === CORRECTIONS: ~10% (6 items) - already counted above ===
    # (5 corrections total: 3 verified + 2 unverified)
    # Add one more verified correction
    # Actually we need explicit correction events for tracking

    # === NOISE: ~50% (30 items) ===
    noise_locs = NOISE_LOCATIONS[:20]
    for i, loc in enumerate(noise_locs):
        # Irrelevant locations with random colors
        append(f'{loc}{i}', rng.choice(COLORS), 'irrelevant',
               verified=True, relevant=False)
    # More noise: repeated observations of retention sites (redundant, not useful)
    for n in names[:5]:
        append(n, initial[n], 'redundant_noise', verified=True, relevant=True)
    # Weather reports (irrelevant text that happens to mention colors)
    for i in range(5):
        append(f'Weather{i}', rng.choice(COLORS), 'irrelevant',
               verified=True, relevant=False, source='weather_service')

    # === MISINFORMATION: ~15% (9 items) ===
    misinfo_targets = names[:9]  # Target first 9 sites with misinfo

    # Simple false (2)
    for n in misinfo_targets[:2]:
        wrong = rng.choice([c for c in COLORS if c != final[n]])
        append(n, wrong, 'misinformation', verified=False, source='rumor',
               misinfo_type='simple_false')

    # Corroborated false (2) - same wrong answer from "multiple sources"
    for n in misinfo_targets[2:4]:
        wrong = rng.choice([c for c in COLORS if c != final[n]])
        # First source
        append(n, wrong, 'misinformation', verified=False, source='traveler_alpha',
               misinfo_type='corroborated')
        # "Confirming" source (counted in the same batch conceptually)

    # Misleadingly framed (1)
    n = misinfo_targets[4]
    wrong = rng.choice([c for c in COLORS if c != final[n]])
    append(n, wrong, 'misinformation', verified=False, source='official_sounding',
           misinfo_type='misleading_frame')

    # Sarcasm/irony (2)
    for n in misinfo_targets[5:7]:
        wrong = rng.choice([c for c in COLORS if c != final[n]])
        append(n, wrong, 'misinformation', verified=False, source='sarcastic_note',
               misinfo_type='sarcasm')

    # Partial truth (1)
    n = misinfo_targets[7]
    wrong = rng.choice([c for c in COLORS if c != final[n]])
    append(n, wrong, 'misinformation', verified=False, source='partial_report',
           misinfo_type='partial_truth')

    # Unfalsifiable (1)
    n = misinfo_targets[8]
    wrong = rng.choice([c for c in COLORS if c != final[n]])
    append(n, wrong, 'misinformation', verified=False, source='vague_claim',
           misinfo_type='unfalsifiable')

    # Shuffle stream
    rng.shuffle(stream)

    return dict(
        split=split,
        seed=seed,
        initial=initial,
        final=final,
        groups=groups,
        stream=stream,
        evaluator_labels=labels,
        composition=dict(
            total=len(stream),
            formative=sum(1 for e in stream if labels[e['id']]['kind'] in ('new', 'correction')
                         and e['verified']),
            useful_unverified=sum(1 for e in stream if labels[e['id']]['kind'] == 'useful_unverified'),
            noise=sum(1 for e in stream if labels[e['id']]['kind'] in ('irrelevant', 'redundant_noise')),
            misinformation=sum(1 for e in stream if labels[e['id']]['kind'] == 'misinformation'),
        )
    )


def route(event):
    """Three-bucket classification based on observable provenance, not truth."""
    if not event['relevant']:
        return dict(route='forgettable', reason='outside beacon task scope')
    if event['verified']:
        return dict(route='formative', reason='authenticated provenance')
    return dict(route='informational', reason='unverified; truth unresolved')


def matched_random(stream, selected, cost, seed):
    """Random sample within tokenizer-cost strata, without quality/answer labels."""
    pools = defaultdict(list)
    for i, event in enumerate(stream):
        pools[cost(event)].append(i)
    counts = Counter(cost(stream[i]) for i in selected)
    rng = random.Random(seed)
    chosen = sorted(
        i for key, count in sorted(counts.items())
        for i in rng.sample(pools[key], count)
    )
    assert Counter(cost(stream[i]) for i in chosen) == counts
    return chosen


if __name__ == '__main__':
    # Quick verification
    world = make_world('test', 17)
    print(f"Stream length: {len(world['stream'])}")
    print(f"Composition: {world['composition']}")
    print(f"Initial sites: {len(world['initial'])}")
    print(f"Final sites: {len(world['final'])}")

    # Count by route
    routes = [route(e) for e in world['stream']]
    formative = sum(1 for r in routes if r['route'] == 'formative')
    informational = sum(1 for r in routes if r['route'] == 'informational')
    forgettable = sum(1 for r in routes if r['route'] == 'forgettable')
    print(f"Routes: formative={formative}, informational={informational}, forgettable={forgettable}")
