@echo off
REM Eco-Loop quick demo (Gemma 4 E2B + demo simulation)
python -m pip install -r requirements.txt
python main.py --mode full --demo
streamlit run dashboard/app.py
