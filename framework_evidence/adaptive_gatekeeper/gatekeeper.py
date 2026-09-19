"""Trainable gatekeeper using DistilBERT for classification.

Option A from spec: fine-tuned small classifier (~250MB)
Input: text of incoming item + current knowledge summary
Output: classification (formative/informational/forgettable) + confidence
"""
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

CLASSIFICATION_LABELS = ['formative', 'informational', 'forgettable']
LABEL_TO_ID = {label: i for i, label in enumerate(CLASSIFICATION_LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(CLASSIFICATION_LABELS)}


@dataclass
class GatekeeperDecision:
    event_id: str
    classification: str
    confidence: float
    probabilities: List[float]


class TrainableGatekeeper(nn.Module):
    """DistilBERT-based classifier for event classification."""

    def __init__(self, model_name: str = 'distilbert-base-uncased', num_labels: int = 3):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

    def classify(self, text: str, knowledge_summary: str = '') -> Tuple[str, float, List[float]]:
        """Classify a single text item."""
        self.eval()
        # Combine item text with knowledge summary
        if knowledge_summary:
            combined = f"[Knowledge] {knowledge_summary} [Item] {text}"
        else:
            combined = text

        inputs = self.tokenizer(
            combined,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True,
        )

        with torch.no_grad():
            if next(self.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            logits = self(inputs['input_ids'], inputs['attention_mask'])
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

        pred_id = max(range(len(probs)), key=probs.__getitem__)
        return ID_TO_LABEL[pred_id], probs[pred_id], probs

    def classify_batch(self, events: List[Dict], knowledge_summary: str = '') -> List[GatekeeperDecision]:
        """Classify a batch of events."""
        decisions = []
        for event in events:
            text = f"Site: {event['name']}, Color: {event['color']}, Source: {event['source']}"
            classification, confidence, probs = self.classify(text, knowledge_summary)
            decisions.append(GatekeeperDecision(
                event_id=event['id'],
                classification=classification,
                confidence=confidence,
                probabilities=probs,
            ))
        return decisions


class GatekeeperTrainer:
    """Trainer for updating gatekeeper from outcome signals."""

    def __init__(self, gatekeeper: TrainableGatekeeper, lr: float = 2e-5):
        self.gatekeeper = gatekeeper
        self.optimizer = torch.optim.AdamW(gatekeeper.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.training_history = []

    def update_from_outcomes(
        self,
        decisions: List[GatekeeperDecision],
        outcomes: Dict[str, Dict],
        threshold: float = 0.02
    ) -> Dict:
        """Update gatekeeper based on outcome signals.

        Args:
            decisions: List of gatekeeper decisions
            outcomes: Dict mapping event_id to outcome info:
                - 'delta': accuracy change after learning this item
                - 'was_useful': whether item improved performance
                - 'was_harmful': whether item degraded performance
            threshold: Minimum delta to count as signal

        Returns:
            Training statistics
        """
        self.gatekeeper.train()

        # Build training examples from outcomes
        positive_examples = []  # Correct classifications
        negative_examples = []  # Incorrect classifications

        for decision in decisions:
            outcome = outcomes.get(decision.event_id)
            if not outcome:
                continue

            delta = outcome.get('delta', 0)
            was_useful = outcome.get('was_useful', False)
            was_harmful = outcome.get('was_harmful', False)

            # Skip weak signals
            if abs(delta) < threshold:
                continue

            # Formative that helped -> reinforce formative
            # Formative that harmed -> should have been informational or forgettable
            # Forgettable that would have helped -> should have been formative
            if decision.classification == 'formative':
                if was_useful and delta > 0:
                    positive_examples.append((decision, 'formative'))
                elif was_harmful and delta < 0:
                    negative_examples.append((decision, 'informational'))
            elif decision.classification == 'forgettable':
                if was_useful:  # Retroactive check showed it would have helped
                    negative_examples.append((decision, 'formative'))

        if not positive_examples and not negative_examples:
            return {'trained': False, 'reason': 'no_strong_signals'}

        # Create training batch
        all_examples = positive_examples + negative_examples
        texts = []
        labels = []

        for decision, target_label in all_examples:
            # Reconstruct text from decision (simplified)
            texts.append(f"Event {decision.event_id}")
            labels.append(LABEL_TO_ID[target_label])

        # Tokenize
        inputs = self.gatekeeper.tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True,
        )
        labels_tensor = torch.tensor(labels)

        if next(self.gatekeeper.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
            labels_tensor = labels_tensor.cuda()

        # Training step
        self.optimizer.zero_grad()
        logits = self.gatekeeper(inputs['input_ids'], inputs['attention_mask'])
        loss = self.criterion(logits, labels_tensor)
        loss.backward()
        self.optimizer.step()

        stats = {
            'trained': True,
            'loss': float(loss.item()),
            'positive_examples': len(positive_examples),
            'negative_examples': len(negative_examples),
        }
        self.training_history.append(stats)

        return stats

    def save(self, path: Path):
        """Save gatekeeper state."""
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.gatekeeper.state_dict(), path / 'gatekeeper.pt')
        (path / 'training_history.json').write_text(
            json.dumps(self.training_history, indent=2),
            encoding='utf-8'
        )

    def load(self, path: Path):
        """Load gatekeeper state."""
        self.gatekeeper.load_state_dict(torch.load(path / 'gatekeeper.pt'))
        history_path = path / 'training_history.json'
        if history_path.exists():
            self.training_history = json.loads(history_path.read_text())


def create_static_gatekeeper():
    """Create a static gatekeeper using verification flags (control arm)."""

    def classify(event: Dict) -> GatekeeperDecision:
        if not event.get('relevant', True):
            classification = 'forgettable'
        elif event.get('verified', False):
            classification = 'formative'
        else:
            classification = 'informational'

        return GatekeeperDecision(
            event_id=event['id'],
            classification=classification,
            confidence=1.0,
            probabilities=[1.0 if i == LABEL_TO_ID[classification] else 0.0
                          for i in range(3)],
        )

    return classify


def initialize_from_previous_experiments(
    gatekeeper: TrainableGatekeeper,
    experiment_paths: List[Path],
    epochs: int = 3,
    lr: float = 2e-5,
) -> Dict:
    """Initialize gatekeeper by training on previous experiment classifications.

    Uses supplied verification flags as initial training data to match
    static gatekeeper baseline.
    """
    from transformers import AutoTokenizer

    # Collect training data from previous experiments
    training_data = []

    for exp_path in experiment_paths:
        # Look for world files with classification data
        for world_file in exp_path.glob('**/world.json'):
            try:
                world = json.loads(world_file.read_text())
                for event in world.get('stream', []):
                    # Determine label from verification flags
                    if not event.get('relevant', True):
                        label = 'forgettable'
                    elif event.get('verified', False):
                        label = 'formative'
                    else:
                        label = 'informational'

                    text = f"Site: {event['name']}, Color: {event['color']}, Source: {event.get('source', 'unknown')}"
                    training_data.append((text, label))
            except Exception:
                continue

    if not training_data:
        return {'initialized': False, 'reason': 'no_training_data'}

    # Train gatekeeper
    gatekeeper.train()
    optimizer = torch.optim.AdamW(gatekeeper.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for text, label in training_data:
            inputs = gatekeeper.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True,
            )
            label_tensor = torch.tensor([LABEL_TO_ID[label]])

            if next(gatekeeper.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
                label_tensor = label_tensor.cuda()

            optimizer.zero_grad()
            logits = gatekeeper(inputs['input_ids'], inputs['attention_mask'])
            loss = criterion(logits, label_tensor)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / len(training_data))

    return {
        'initialized': True,
        'training_examples': len(training_data),
        'epochs': epochs,
        'final_loss': losses[-1] if losses else None,
    }


if __name__ == '__main__':
    # Test gatekeeper creation
    print("Creating trainable gatekeeper...")
    gatekeeper = TrainableGatekeeper()

    # Test classification
    test_event = {
        'id': 'test001',
        'name': 'TestSite',
        'color': 'blue',
        'source': 'registry',
        'relevant': True,
        'verified': True,
    }

    print("\nTesting classification...")
    text = f"Site: {test_event['name']}, Color: {test_event['color']}, Source: {test_event['source']}"
    classification, confidence, probs = gatekeeper.classify(text)
    print(f"Classification: {classification}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Probabilities: {[f'{p:.3f}' for p in probs]}")

    # Test static gatekeeper
    print("\nTesting static gatekeeper...")
    static_classify = create_static_gatekeeper()
    decision = static_classify(test_event)
    print(f"Static classification: {decision.classification}")
