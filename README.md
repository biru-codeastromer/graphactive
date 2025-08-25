# GraphActive

Budgeted active learning on graphs. A tiny benchmark showing community-aware sampling beats naive uncertainty sampling in early rounds for node classification on a Cora-like synthetic SBM.

## Quickstart
```bash
python -m venv .venv
. .venv/bin/activate    # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
make reproduce
make plot
make test
```

## Streamlit
```bash
make app
```

