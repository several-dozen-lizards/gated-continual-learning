"""Verify sanitized evidence bytes; adapter assets have separate checksums."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'PUBLICATION_MANIFEST.json').read_text(encoding='utf-8'))
for entry in manifest['files']:
    path=root/entry['path']
    assert path.is_file(),entry['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['published_sha256'],entry['path']
print('Verified',len(manifest['files']),'published study files.')
framework=root/'FRAMEWORK_EVIDENCE_MANIFEST.json'
if framework.exists():
    additional=json.loads(framework.read_text(encoding='utf-8'))
    for entry in additional['files']:
        path=root/entry['path']
        assert path.is_file(),entry['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['published_sha256'],entry['path']
    print('Verified',len(additional['files']),'framework evidence files.')
print('Historical hashes are preserved as provenance; sanitized bytes use the published checksums.')
print('Model reload validation requires base weights and the separate checkpoint assets.')
