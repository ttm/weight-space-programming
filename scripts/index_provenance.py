"""Regenerate an evidence-aware provenance index from exact candidate bytes."""
from pathlib import Path
import json
from wsp.provenance import index

if __name__=='__main__':
    data=index(Path('.'))
    destination=Path('results/provenance_index.json')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf8')
    print(f"Indexed {data['total']} artifacts; {data['structured']} with structured provenance.")
