"""Outcome signal collection and gatekeeper update loop.

Implements the Tier 2 feedback mechanism:
1. Record gatekeeper decisions
2. Run primary model learning
3. Measure performance delta
4. Compute per-decision outcome signals
5. Update gatekeeper on accumulated signals
"""
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Callable
from pathlib import Path

from gatekeeper import GatekeeperDecision, LABEL_TO_ID


@dataclass
class OutcomeSignal:
    """Outcome signal for a single gatekeeper decision."""
    event_id: str
    classification: str
    accuracy_before: float
    accuracy_after: float
    delta: float
    was_useful: bool
    was_harmful: bool
    retroactive_check: Optional[bool] = None  # For forgettable items


@dataclass
class BatchOutcome:
    """Outcomes for a batch of gatekeeper updates."""
    batch_id: int
    decisions: List[GatekeeperDecision]
    signals: List[OutcomeSignal]
    gatekeeper_accuracy_before: float
    gatekeeper_accuracy_after: float
    primary_accuracy_before: float
    primary_accuracy_after: float


class FeedbackLoop:
    """Manages outcome signal collection and gatekeeper updates."""

    def __init__(
        self,
        batch_size: int = 50,
        delta_threshold: float = 0.02,
        retroactive_sample_rate: float = 0.15,
        seed: int = 17,
    ):
        self.batch_size = batch_size
        self.delta_threshold = delta_threshold
        self.retroactive_sample_rate = retroactive_sample_rate
        self.rng = random.Random(seed)

        self.current_batch: List[GatekeeperDecision] = []
        self.current_events: List[Dict] = []
        self.batch_history: List[BatchOutcome] = []
        self.batch_id = 0

    def record_decision(self, decision: GatekeeperDecision, event: Dict):
        """Record a gatekeeper decision for later outcome evaluation."""
        self.current_batch.append(decision)
        self.current_events.append(event)

    def is_batch_complete(self) -> bool:
        """Check if current batch is ready for processing."""
        return len(self.current_batch) >= self.batch_size

    def compute_outcomes(
        self,
        evaluate_fn: Callable[[], float],
        train_fn: Callable[[List[Dict]], None],
        retroactive_train_fn: Optional[Callable[[Dict], float]] = None,
    ) -> BatchOutcome:
        """Compute outcome signals for current batch.

        Args:
            evaluate_fn: Function that returns current model accuracy
            train_fn: Function that trains model on a list of events
            retroactive_train_fn: Optional function to test training on a single
                                 event and return accuracy delta (for forgettable checks)

        Returns:
            BatchOutcome with all signals
        """
        # Measure accuracy before this batch
        accuracy_before = evaluate_fn()

        # Separate events by classification
        formative_events = []
        informational_events = []
        forgettable_events = []

        for decision, event in zip(self.current_batch, self.current_events):
            if decision.classification == 'formative':
                formative_events.append((decision, event))
            elif decision.classification == 'informational':
                informational_events.append((decision, event))
            else:
                forgettable_events.append((decision, event))

        # Train on formative events
        if formative_events:
            train_fn([event for _, event in formative_events])

        # Measure accuracy after training
        accuracy_after = evaluate_fn()

        # Build outcome signals
        signals = []
        overall_delta = accuracy_after - accuracy_before

        # For formative events: attribute delta proportionally
        # (simplified - in practice would need per-item measurement)
        for decision, event in formative_events:
            per_item_delta = overall_delta / len(formative_events) if formative_events else 0
            signals.append(OutcomeSignal(
                event_id=decision.event_id,
                classification='formative',
                accuracy_before=accuracy_before,
                accuracy_after=accuracy_after,
                delta=per_item_delta,
                was_useful=per_item_delta > self.delta_threshold,
                was_harmful=per_item_delta < -self.delta_threshold,
            ))

        # For informational events: no direct signal (no training occurred)
        for decision, event in informational_events:
            signals.append(OutcomeSignal(
                event_id=decision.event_id,
                classification='informational',
                accuracy_before=accuracy_before,
                accuracy_after=accuracy_after,
                delta=0,
                was_useful=False,
                was_harmful=False,
            ))

        # For forgettable events: sample for retroactive checking
        if retroactive_train_fn and forgettable_events:
            sample_size = max(1, int(len(forgettable_events) * self.retroactive_sample_rate))
            sampled = self.rng.sample(forgettable_events, min(sample_size, len(forgettable_events)))

            for decision, event in forgettable_events:
                if (decision, event) in sampled:
                    # Retroactive check: would this have helped?
                    retro_delta = retroactive_train_fn(event)
                    signals.append(OutcomeSignal(
                        event_id=decision.event_id,
                        classification='forgettable',
                        accuracy_before=accuracy_before,
                        accuracy_after=accuracy_after,
                        delta=0,
                        was_useful=retro_delta > self.delta_threshold,
                        was_harmful=False,
                        retroactive_check=retro_delta > self.delta_threshold,
                    ))
                else:
                    signals.append(OutcomeSignal(
                        event_id=decision.event_id,
                        classification='forgettable',
                        accuracy_before=accuracy_before,
                        accuracy_after=accuracy_after,
                        delta=0,
                        was_useful=False,
                        was_harmful=False,
                        retroactive_check=None,
                    ))
        else:
            for decision, event in forgettable_events:
                signals.append(OutcomeSignal(
                    event_id=decision.event_id,
                    classification='forgettable',
                    accuracy_before=accuracy_before,
                    accuracy_after=accuracy_after,
                    delta=0,
                    was_useful=False,
                    was_harmful=False,
                ))

        # Create batch outcome
        outcome = BatchOutcome(
            batch_id=self.batch_id,
            decisions=list(self.current_batch),
            signals=signals,
            gatekeeper_accuracy_before=0,  # Will be filled by caller
            gatekeeper_accuracy_after=0,
            primary_accuracy_before=accuracy_before,
            primary_accuracy_after=accuracy_after,
        )

        self.batch_history.append(outcome)
        self.batch_id += 1

        # Clear current batch
        self.current_batch = []
        self.current_events = []

        return outcome

    def get_outcome_dict(self, outcome: BatchOutcome) -> Dict[str, Dict]:
        """Convert BatchOutcome to dict for gatekeeper trainer."""
        return {
            signal.event_id: {
                'delta': signal.delta,
                'was_useful': signal.was_useful,
                'was_harmful': signal.was_harmful,
            }
            for signal in outcome.signals
        }

    def save(self, path: Path):
        """Save feedback loop state."""
        path.mkdir(parents=True, exist_ok=True)
        history = [
            {
                'batch_id': outcome.batch_id,
                'decisions': [asdict(d) for d in outcome.decisions],
                'signals': [asdict(s) for s in outcome.signals],
                'gatekeeper_accuracy_before': outcome.gatekeeper_accuracy_before,
                'gatekeeper_accuracy_after': outcome.gatekeeper_accuracy_after,
                'primary_accuracy_before': outcome.primary_accuracy_before,
                'primary_accuracy_after': outcome.primary_accuracy_after,
            }
            for outcome in self.batch_history
        ]
        (path / 'feedback_history.json').write_text(
            json.dumps(history, indent=2),
            encoding='utf-8'
        )

    def load(self, path: Path):
        """Load feedback loop state."""
        history_path = path / 'feedback_history.json'
        if history_path.exists():
            data = json.loads(history_path.read_text())
            self.batch_history = [
                BatchOutcome(
                    batch_id=item['batch_id'],
                    decisions=[GatekeeperDecision(**d) for d in item['decisions']],
                    signals=[OutcomeSignal(**s) for s in item['signals']],
                    gatekeeper_accuracy_before=item['gatekeeper_accuracy_before'],
                    gatekeeper_accuracy_after=item['gatekeeper_accuracy_after'],
                    primary_accuracy_before=item['primary_accuracy_before'],
                    primary_accuracy_after=item['primary_accuracy_after'],
                )
                for item in data
            ]
            self.batch_id = len(self.batch_history)


def compute_gatekeeper_accuracy(
    decisions: List[GatekeeperDecision],
    ground_truth: Dict[str, str],
) -> float:
    """Compute gatekeeper classification accuracy against ground truth.

    Args:
        decisions: Gatekeeper decisions
        ground_truth: Dict mapping event_id to correct classification

    Returns:
        Accuracy (0-1)
    """
    if not decisions:
        return 0.0

    correct = sum(
        1 for d in decisions
        if ground_truth.get(d.event_id) == d.classification
    )
    return correct / len(decisions)


def build_ground_truth(events: List[Dict], labels: Dict[str, Dict]) -> Dict[str, str]:
    """Build ground truth classifications from event labels.

    Uses the same logic as static gatekeeper but with oracle knowledge
    of what would have been useful.
    """
    truth = {}
    for event in events:
        label = labels.get(event['id'], {})

        # Forgettable: irrelevant to task
        if not event.get('relevant', True):
            truth[event['id']] = 'forgettable'
        # Formative: useful updates that should be learned
        elif label.get('useful', False) and not label.get('kind') == 'misinformation':
            truth[event['id']] = 'formative'
        # Informational: relevant but not for weight updates
        # (includes misinformation, unverified claims)
        else:
            truth[event['id']] = 'informational'

    return truth


if __name__ == '__main__':
    # Test feedback loop
    print("Testing FeedbackLoop...")

    loop = FeedbackLoop(batch_size=5)

    # Create mock decisions
    for i in range(5):
        decision = GatekeeperDecision(
            event_id=f'e{i:04}',
            classification='formative' if i < 3 else 'forgettable',
            confidence=0.9,
            probabilities=[0.9, 0.05, 0.05],
        )
        event = {'id': f'e{i:04}', 'name': f'Site{i}', 'color': 'blue', 'source': 'registry'}
        loop.record_decision(decision, event)

    print(f"Batch complete: {loop.is_batch_complete()}")

    # Compute outcomes with mock functions
    def mock_evaluate():
        return 0.8

    def mock_train(events):
        print(f"Training on {len(events)} events")

    outcome = loop.compute_outcomes(mock_evaluate, mock_train)
    print(f"Batch {outcome.batch_id}: {len(outcome.signals)} signals")
    print(f"Primary accuracy: {outcome.primary_accuracy_before:.2f} -> {outcome.primary_accuracy_after:.2f}")
