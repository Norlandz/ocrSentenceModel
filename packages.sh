pip install ^
  "numpy<2.0" ^
  pandas ^
  itables ^
  matplotlib ^
  plotly ^
  seaborn ^
  scikit-learn ^
  tqdm ^
  ipykernel ^
  ipywidgets ^
  nbformat


# not_working; use global, not venv
pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# pip install torchtext
# pip install gdown

pip install datasets

pip install "fastapi[standard]"
# pip install fastapi

pip install pydantic

pip install "uvicorn[standard]"
# pip install fastapi uvicorn

pip install fastapi-cors

pip install pytest

########################################

# fastapi run ./src/main.py
# uvicorn main:app --host 0.0.0.0 --port 80
# uvicorn main:app --host 127.0.0.1 --port 8000
uvicorn src.main:app --host 127.0.0.1 --port 8000