@echo off
echo [1/4] Tao virtual environment...
python -m venv venv

echo [2/4] Kich hoat venv...
call venv\Scripts\activate.bat

echo [3/4] Cai CLIP...
pip install git+https://github.com/openai/CLIP.git

echo [4/4] Cai cac thu vien con lai...
pip install -r requirements.txt

echo.
echo  Xong! Chay "venv\Scripts\activate.bat" de bat dau lam viec.
pause
