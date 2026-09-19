"""Extended evidence stream with difficulty progression for adaptive gatekeeper.

300+ items with misinformation difficulty tiers:
- Tier 1 (Easy): positions 1-100 — simple false claims
- Tier 2 (Medium): positions 50-300 — corroborated, shifting narratives
- Tier 3 (Hard): positions 150+ — misleading frames, sarcasm, partial truths

The difficulty overlap is intentional — tests whether gatekeeper adaptation
keeps pace with increasing adversarial sophistication.
"""
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

COLORS = ('red', 'blue', 'green', 'gold')

# Large name pool for 300+ item stream
SITE_NAMES = [
    # Original test names (20)
    'Amber', 'Beryl', 'Coral', 'Flint', 'Garnet', 'Jade', 'Jasper', 'Onyx',
    'Opal', 'Pearl', 'Quartz', 'Ruby', 'Agate', 'Cobalt', 'Ember', 'Frost',
    'Granite', 'Haze', 'Ivory', 'Kyanite',
    # Extended names (30 more)
    'Larimar', 'Malachite', 'Nephrite', 'Obsidian', 'Peridot', 'Rhodonite',
    'Sapphire', 'Topaz', 'Turquoise', 'Zircon', 'Aventurine', 'Bloodstone',
    'Carnelian', 'Dolomite', 'Emerald', 'Fluorite', 'Galena', 'Hematite',
    'Iolite', 'Jadeite', 'Kunzite', 'Labradorite', 'Moonstone', 'Nuummite',
    'Orthoclase', 'Pyrite', 'Quartzite', 'Rhodochrosite', 'Sodalite', 'Tanzanite',
]

# Noise location prefixes
NOISE_PREFIXES = [
    'Outpost', 'Station', 'Waypoint', 'Depot', 'Shelter', 'Camp', 'Refuge',
    'Haven', 'Retreat', 'Clearing', 'Crossing', 'Junction', 'Summit', 'Valley',
    'Ridge', 'Mesa', 'Basin', 'Delta', 'Cove', 'Inlet', 'Harbor', 'Port',
]

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

# Misinformation difficulty tiers
MISINFO_TIERS = {
    'easy': {
        'types': ['simple_false', 'obvious_wrong'],
        'position_range': (0, 100),
    },
    'medium': {
        'types': ['corroborated', 'shifting_narrative', 'subtle_contradiction'],
        'position_range': (50, 300),
    },
    'hard': {
        'types': ['misleading_frame', 'sarcasm', 'partial_truth', 'unfalsifiable', 'false_correction'],
        'position_range': (150, 500),
    },
}


@dataclass
class StreamEvent:
    id: str
    position: int  # Original position in stream before shuffle
    name: str
    color: str
    relevant: bool
    verified: bool
    source: str
    kind: str
    useful: bool
    misinfo_type: Optional[str] = None
    difficulty_tier: Optional[str] = None
    corroboration_group: Optional[str] = None  # For tracking corroborated claims


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def make_extended_stream(seed: int, target_size: int = 300) -> Dict:
    """Generate extended stream with difficulty progression."""
    rng = random.Random(seed)

    # Use all available site names
    names = list(SITE_NAMES[:50])  # 50 sites for beacon facts

    # Initial facts: first 20 sites
    initial_colors = (list(COLORS) * 5)[:20]
    rng.shuffle(initial_colors)
    initial = dict(zip(names[:20], initial_colors))

    # Final state: corrections for sites 10-20, new facts for sites 20-50
    final = dict(initial)
    for n in names[10:20]:  # 10 corrections
        final[n] = rng.choice([c for c in COLORS if c != initial[n]])
    new_colors = (list(COLORS) * 8)[:30]
    rng.shuffle(new_colors)
    for i, n in enumerate(names[20:50]):  # 30 new sites
        final[n] = new_colors[i]

    # Build groups
    groups = {}
    for n in names[:10]:
        groups[n] = 'retention'
    for n in names[10:20]:
        groups[n] = 'correction'
    for n in names[20:50]:
        groups[n] = 'new'

    events: List[StreamEvent] = []
    position = 0

    def add_event(name, color, kind, verified=False, relevant=True, source='registry',
                  misinfo_type=None, difficulty_tier=None, corroboration_group=None):
        nonlocal position
        events.append(StreamEvent(
            id=f'e{len(events):04}',
            position=position,
            name=name,
            color=color,
            relevant=relevant,
            verified=verified,
            source=source,
            kind=kind,
            useful=kind in ('new', 'correction', 'useful_unverified'),
            misinfo_type=misinfo_type,
            difficulty_tier=difficulty_tier,
            corroboration_group=corroboration_group,
        ))
        position += 1

    # === FORMATIVE: ~15% (45 items) ===
    # Verified new facts (30)
    for n in names[20:50]:
        add_event(n, final[n], 'new', verified=True, source='registry')

    # Verified corrections (10)
    for n in names[10:20]:
        add_event(n, final[n], 'correction', verified=True, source='registry')

    # Additional verified facts (5)
    for n in rng.sample(names[:10], 5):
        add_event(n, initial[n], 'retention_verified', verified=True, source='registry')

    # === USEFUL-UNVERIFIED: ~10% (30 items) ===
    for _ in range(30):
        n = rng.choice(names)
        add_event(n, final.get(n, rng.choice(COLORS)), 'useful_unverified',
                  verified=False, source='traveler')

    # === NOISE: ~45% (135 items) ===
    # Irrelevant locations
    for i in range(80):
        prefix = rng.choice(NOISE_PREFIXES)
        add_event(f'{prefix}{i}', rng.choice(COLORS), 'irrelevant',
                  verified=True, relevant=False)

    # Redundant observations
    for _ in range(30):
        n = rng.choice(names[:20])
        add_event(n, initial.get(n, final[n]), 'redundant_noise',
                  verified=True, relevant=True)

    # Weather/misc noise
    for i in range(25):
        add_event(f'Weather{i}', rng.choice(COLORS), 'irrelevant',
                  verified=True, relevant=False, source='weather_service')

    # === MISINFORMATION: ~20% (60 items) with difficulty progression ===

    # Tier 1 - Easy (positions 0-100): 20 items
    misinfo_targets_easy = rng.sample(names, 10)
    for n in misinfo_targets_easy:
        wrong = rng.choice([c for c in COLORS if c != final.get(n, 'red')])
        # Simple false
        add_event(n, wrong, 'misinformation', source='rumor',
                  misinfo_type='simple_false', difficulty_tier='easy')
        # Obviously wrong
        add_event(n, wrong, 'misinformation', source='stranger',
                  misinfo_type='obvious_wrong', difficulty_tier='easy')

    # Tier 2 - Medium (positions 50-300): 20 items
    misinfo_targets_medium = rng.sample(names, 10)
    corr_group = 0
    for n in misinfo_targets_medium:
        wrong = rng.choice([c for c in COLORS if c != final.get(n, 'red')])
        # Corroborated false (multiple sources agree)
        group_id = f'corr_{corr_group}'
        add_event(n, wrong, 'misinformation', source='traveler_alpha',
                  misinfo_type='corroborated', difficulty_tier='medium',
                  corroboration_group=group_id)
        add_event(n, wrong, 'misinformation', source='traveler_beta',
                  misinfo_type='corroborated', difficulty_tier='medium',
                  corroboration_group=group_id)
        corr_group += 1

    # Tier 3 - Hard (positions 150+): 20 items
    misinfo_targets_hard = rng.sample(names, 10)
    for i, n in enumerate(misinfo_targets_hard):
        wrong = rng.choice([c for c in COLORS if c != final.get(n, 'red')])
        # Distribute across hard types
        hard_types = ['misleading_frame', 'sarcasm', 'partial_truth',
                      'unfalsifiable', 'false_correction']
        mtype = hard_types[i % len(hard_types)]
        source = {
            'misleading_frame': 'official_sounding',
            'sarcasm': 'sarcastic_note',
            'partial_truth': 'partial_report',
            'unfalsifiable': 'vague_claim',
            'false_correction': 'correction_notice',
        }.get(mtype, 'unknown')

        add_event(n, wrong, 'misinformation', source=source,
                  misinfo_type=mtype, difficulty_tier='hard')
        # Add a second hard item
        add_event(n, wrong, 'misinformation', source=source + '_2',
                  misinfo_type=mtype, difficulty_tier='hard')

    # === CORRECTIONS: ~10% (30 items) ===
    # These are legitimate corrections, not false corrections
    for _ in range(30):
        n = rng.choice(names[10:20])  # Sites that have corrections
        add_event(n, final[n], 'correction', verified=True, source='registry_update')

    # Shuffle stream while respecting difficulty position ranges
    # First, assign target positions based on difficulty tier
    easy_events = [e for e in events if e.difficulty_tier == 'easy']
    medium_events = [e for e in events if e.difficulty_tier == 'medium']
    hard_events = [e for e in events if e.difficulty_tier == 'hard']
    other_events = [e for e in events if e.difficulty_tier is None]

    # Create position pools
    final_stream = [None] * len(events)
    all_positions = list(range(len(events)))
    rng.shuffle(all_positions)

    # Place easy events in positions 0-100
    easy_positions = [p for p in all_positions if p < 100][:len(easy_events)]
    for event, pos in zip(easy_events, easy_positions):
        final_stream[pos] = event
        all_positions.remove(pos)

    # Place medium events in positions 50-300
    medium_positions = [p for p in all_positions if 50 <= p < 300][:len(medium_events)]
    for event, pos in zip(medium_events, medium_positions):
        final_stream[pos] = event
        all_positions.remove(pos)

    # Place hard events in positions 150+
    hard_positions = [p for p in all_positions if p >= 150][:len(hard_events)]
    for event, pos in zip(hard_events, hard_positions):
        final_stream[pos] = event
        all_positions.remove(pos)

    # Place remaining events in remaining positions
    rng.shuffle(other_events)
    for event, pos in zip(other_events, all_positions):
        final_stream[pos] = event

    # Fill any None spots (shouldn't happen, but safety)
    remaining = [e for e in events if e not in final_stream]
    for i, slot in enumerate(final_stream):
        if slot is None and remaining:
            final_stream[i] = remaining.pop(0)

    # Update positions
    for i, event in enumerate(final_stream):
        if event:
            event.position = i

    # Convert to dicts for JSON serialization
    stream_dicts = [asdict(e) for e in final_stream if e]

    # Compute composition
    composition = {
        'total': len(stream_dicts),
        'formative': sum(1 for e in stream_dicts if e['kind'] in ('new', 'correction', 'retention_verified') and e['verified']),
        'useful_unverified': sum(1 for e in stream_dicts if e['kind'] == 'useful_unverified'),
        'noise': sum(1 for e in stream_dicts if e['kind'] in ('irrelevant', 'redundant_noise')),
        'misinformation': sum(1 for e in stream_dicts if e['kind'] == 'misinformation'),
        'corrections': sum(1 for e in stream_dicts if e['kind'] == 'correction'),
        'by_difficulty': {
            'easy': sum(1 for e in stream_dicts if e.get('difficulty_tier') == 'easy'),
            'medium': sum(1 for e in stream_dicts if e.get('difficulty_tier') == 'medium'),
            'hard': sum(1 for e in stream_dicts if e.get('difficulty_tier') == 'hard'),
        },
    }

    return dict(
        seed=seed,
        initial=initial,
        final=final,
        groups=groups,
        stream=stream_dicts,
        composition=composition,
    )


def route(event: dict) -> dict:
    """Three-bucket classification based on observable provenance."""
    if not event['relevant']:
        return dict(route='forgettable', reason='outside beacon task scope')
    if event['verified']:
        return dict(route='formative', reason='authenticated provenance')
    return dict(route='informational', reason='unverified; truth unresolved')


if __name__ == '__main__':
    # Test stream generation
    stream = make_extended_stream(17)
    print(f"Stream length: {len(stream['stream'])}")
    print(f"Composition: {json.dumps(stream['composition'], indent=2)}")
    print(f"Initial sites: {len(stream['initial'])}")
    print(f"Final sites: {len(stream['final'])}")

    # Check difficulty distribution by position
    easy_pos = [e['position'] for e in stream['stream'] if e.get('difficulty_tier') == 'easy']
    medium_pos = [e['position'] for e in stream['stream'] if e.get('difficulty_tier') == 'medium']
    hard_pos = [e['position'] for e in stream['stream'] if e.get('difficulty_tier') == 'hard']

    print(f"\nDifficulty positions:")
    print(f"  Easy: {min(easy_pos) if easy_pos else 'N/A'}-{max(easy_pos) if easy_pos else 'N/A'} ({len(easy_pos)} items)")
    print(f"  Medium: {min(medium_pos) if medium_pos else 'N/A'}-{max(medium_pos) if medium_pos else 'N/A'} ({len(medium_pos)} items)")
    print(f"  Hard: {min(hard_pos) if hard_pos else 'N/A'}-{max(hard_pos) if hard_pos else 'N/A'} ({len(hard_pos)} items)")

    # Route statistics
    routes = [route(e) for e in stream['stream']]
    print(f"\nRoutes:")
    print(f"  Formative: {sum(1 for r in routes if r['route'] == 'formative')}")
    print(f"  Informational: {sum(1 for r in routes if r['route'] == 'informational')}")
    print(f"  Forgettable: {sum(1 for r in routes if r['route'] == 'forgettable')}")
