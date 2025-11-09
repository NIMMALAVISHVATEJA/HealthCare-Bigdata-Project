python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python generate_data.py
python preprocess.py
python train_model.py
streamlit run app.py
