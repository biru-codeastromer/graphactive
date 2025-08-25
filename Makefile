PY=python

reproduce:
	$(PY) -m ga.cli reproduce

plot:
	$(PY) -m ga.cli plot

test:
	pytest -q

app:
	streamlit run ga/streamlit_app.py