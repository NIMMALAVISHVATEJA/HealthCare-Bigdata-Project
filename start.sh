#!/bin/bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python generate_data.py
python preprocess.py
python train_model.py
streamlit run app.py
