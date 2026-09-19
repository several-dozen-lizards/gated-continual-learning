"""Meta-calibration: Primary model reviews gatekeeper decisions.

Tier 3 of the hierarchy:
- Runs every M gatekeeper update cycles
- Primary model assesses past decisions from current knowledge state
- Calibration signals adjust gatekeeper thresholds or training set
"""
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Callable
from pathlib import Path

from gatekeeper import GatekeeperDecision, LABEL_TO_ID


@dataclass
class CalibrationSignal:
    """Calibration signal from primary model review."""
    event_id: str
    original_classification: str
    suggested_classification: str
    reasoning: str
    confidence: float


@dataclass
class CalibrationCycle:
    """Results from one meta-calibration cycle."""
    cycle_id: int
    batches_reviewed: List[int]
    signals: List[CalibrationSignal]
    decisions_reviewed: int
    corrections_suggested: int
    gatekeeper_updated: bool


class MetaCalibrator:
    """Manages meta-calibration cycles using primary model review."""

    def __init__(
        self,
        cycle_frequency: int = 5,  # Every M gatekeeper updates
        sample_size: int = 20,     # Decisions to review per cycle
        seed: int = 17,
    ):
        self.cycle_frequency = cycle_frequency
        self.sample_size = sample_size
        self.rng = __import__('random').Random(seed)

        self.pending_batches: List[int] = []
        self.pending_decisions: List[GatekeeperDecision] = []
        self.pending_events: List[Dict] = []
        self.calibration_history: List[CalibrationCycle] = []
        self.cycle_id = 0

    def record_batch(self, batch_id: int, decisions: List[GatekeeperDecision], events: List[Dict]):
        """Record a batch for later review."""
        self.pending_batches.append(batch_id)
        self.pending_decisions.extend(decisions)
        self.pending_events.extend(events)

    def is_cycle_due(self) -> bool:
        """Check if meta-calibration cycle is due."""
        return len(self.pending_batches) >= self.cycle_frequency

    def build_review_prompt(
        self,
        decisions: List[GatekeeperDecision],
        events: List[Dict],
        current_knowledge: Dict[str, str],
    ) -> str:
        """Build a structured prompt for primary model review.

        Args:
            decisions: Sample of gatekeeper decisions to review
            events: Corresponding events
            current_knowledge: Dict mapping site names to learned colors

        Returns:
            Structured prompt for primary model
        """
        # Build knowledge summary
        knowledge_lines = [
            f"- {name}: {color}"
            for name, color in sorted(current_knowledge.items())
        ]
        knowledge_section = "\n".join(knowledge_lines) if knowledge_lines else "(no learned facts)"

        # Build decision list
        decision_lines = []
        for decision, event in zip(decisions, events):
            decision_lines.append(
                f"- [{decision.classification.upper()}] Site '{event['name']}' has color '{event['color']}' "
                f"(source: {event.get('source', 'unknown')}, confidence: {decision.confidence:.2f})"
            )
        decisions_section = "\n".join(decision_lines)

        prompt = f"""You have learned the following facts about beacon colors:
{knowledge_section}

The gatekeeper classified the following items. Review each classification:

{decisions_section}

For each item, assess:
1. FORGETTABLE items: From your current knowledge, would any have been useful to learn?
2. FORMATIVE items: Did any seem to hurt your knowledge or contradict established facts?
3. INFORMATIONAL items: Should any have been promoted to formative or demoted to forgettable?

Respond with a JSON list of corrections needed:
[{{"event_id": "...", "current": "formative/informational/forgettable", "suggested": "...", "reason": "..."}}]

Only include items that need correction. Return [] if all classifications were appropriate."""

        return prompt

    def parse_review_response(self, response: str) -> List[CalibrationSignal]:
        """Parse primary model's review response into calibration signals."""
        signals = []

        # Try to extract JSON from response
        try:
            # Look for JSON array in response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                corrections = json.loads(json_match.group())
                for item in corrections:
                    signals.append(CalibrationSignal(
                        event_id=item.get('event_id', ''),
                        original_classification=item.get('current', ''),
                        suggested_classification=item.get('suggested', ''),
                        reasoning=item.get('reason', ''),
                        confidence=0.8,  # Default confidence for model suggestions
                    ))
        except (json.JSONDecodeError, AttributeError):
            # If parsing fails, no corrections suggested
            pass

        return signals

    def run_calibration(
        self,
        review_fn: Callable[[str], str],
        current_knowledge: Dict[str, str],
        gatekeeper_update_fn: Optional[Callable[[List[CalibrationSignal]], Dict]] = None,
    ) -> CalibrationCycle:
        """Run a meta-calibration cycle.

        Args:
            review_fn: Function that takes a prompt and returns model response
            current_knowledge: Current learned facts (site -> color)
            gatekeeper_update_fn: Optional function to update gatekeeper from signals

        Returns:
            CalibrationCycle with results
        """
        # Sample decisions to review
        sample_indices = self.rng.sample(
            range(len(self.pending_decisions)),
            min(self.sample_size, len(self.pending_decisions))
        )
        sampled_decisions = [self.pending_decisions[i] for i in sample_indices]
        sampled_events = [self.pending_events[i] for i in sample_indices]

        # Build and send review prompt
        prompt = self.build_review_prompt(sampled_decisions, sampled_events, current_knowledge)
        response = review_fn(prompt)

        # Parse response
        signals = self.parse_review_response(response)

        # Update gatekeeper if function provided
        gatekeeper_updated = False
        if gatekeeper_update_fn and signals:
            update_result = gatekeeper_update_fn(signals)
            gatekeeper_updated = update_result.get('updated', False)

        # Create cycle record
        cycle = CalibrationCycle(
            cycle_id=self.cycle_id,
            batches_reviewed=list(self.pending_batches),
            signals=signals,
            decisions_reviewed=len(sampled_decisions),
            corrections_suggested=len(signals),
            gatekeeper_updated=gatekeeper_updated,
        )

        self.calibration_history.append(cycle)
        self.cycle_id += 1

        # Clear pending
        self.pending_batches = []
        self.pending_decisions = []
        self.pending_events = []

        return cycle

    def save(self, path: Path):
        """Save calibrator state."""
        path.mkdir(parents=True, exist_ok=True)
        history = [
            {
                'cycle_id': cycle.cycle_id,
                'batches_reviewed': cycle.batches_reviewed,
                'signals': [asdict(s) for s in cycle.signals],
                'decisions_reviewed': cycle.decisions_reviewed,
                'corrections_suggested': cycle.corrections_suggested,
                'gatekeeper_updated': cycle.gatekeeper_updated,
            }
            for cycle in self.calibration_history
        ]
        (path / 'calibration_history.json').write_text(
            json.dumps(history, indent=2),
            encoding='utf-8'
        )

    def load(self, path: Path):
        """Load calibrator state."""
        history_path = path / 'calibration_history.json'
        if history_path.exists():
            data = json.loads(history_path.read_text())
            self.calibration_history = [
                CalibrationCycle(
                    cycle_id=item['cycle_id'],
                    batches_reviewed=item['batches_reviewed'],
                    signals=[CalibrationSignal(**s) for s in item['signals']],
                    decisions_reviewed=item['decisions_reviewed'],
                    corrections_suggested=item['corrections_suggested'],
                    gatekeeper_updated=item['gatekeeper_updated'],
                )
                for item in data
            ]
            self.cycle_id = len(self.calibration_history)


def apply_calibration_to_gatekeeper(
    gatekeeper,
    trainer,
    signals: List[CalibrationSignal],
) -> Dict:
    """Apply calibration signals to gatekeeper.

    This adds the calibration examples to the gatekeeper's training set
    and performs a small update.
    """
    if not signals:
        return {'updated': False, 'reason': 'no_signals'}

    # Convert signals to training examples
    # This is a simplified version - full implementation would
    # retrieve actual event text
    import torch

    gatekeeper.train()
    optimizer = torch.optim.AdamW(gatekeeper.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0
    for signal in signals:
        if signal.suggested_classification not in LABEL_TO_ID:
            continue

        # Create synthetic training example
        text = f"Calibration example {signal.event_id}"
        inputs = gatekeeper.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True,
        )
        label = torch.tensor([LABEL_TO_ID[signal.suggested_classification]])

        if next(gatekeeper.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
            label = label.cuda()

        optimizer.zero_grad()
        logits = gatekeeper(inputs['input_ids'], inputs['attention_mask'])
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return {
        'updated': True,
        'signals_applied': len(signals),
        'mean_loss': total_loss / len(signals) if signals else 0,
    }


if __name__ == '__main__':
    # Test meta-calibrator
    print("Testing MetaCalibrator...")

    calibrator = MetaCalibrator(cycle_frequency=2, sample_size=3)

    # Add mock batches
    for batch_id in range(2):
        decisions = [
            GatekeeperDecision(f'e{batch_id}_{i}', 'formative', 0.9, [0.9, 0.05, 0.05])
            for i in range(5)
        ]
        events = [
            {'id': f'e{batch_id}_{i}', 'name': f'Site{i}', 'color': 'blue', 'source': 'registry'}
            for i in range(5)
        ]
        calibrator.record_batch(batch_id, decisions, events)

    print(f"Cycle due: {calibrator.is_cycle_due()}")

    # Run calibration with mock review
    def mock_review(prompt):
        return '[]'  # No corrections

    current_knowledge = {'Site0': 'blue', 'Site1': 'red'}
    cycle = calibrator.run_calibration(mock_review, current_knowledge)

    print(f"Cycle {cycle.cycle_id}: reviewed {cycle.decisions_reviewed}, suggested {cycle.corrections_suggested}")
