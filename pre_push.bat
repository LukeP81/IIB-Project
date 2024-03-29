@echo off
echo Requirements ...
pip install wheel
pip install -r requirements.txt
pip install flake8
pip install pylint
echo ... Flake8 ...
python -m flake8 . --count --show-source --statistics --max-line-length 82 --exclude ./oak/,tests/test_oak/,venv/
echo ... Pylint ...
python -m pylint --recursive=y .
echo Pytest ...
python -m pytest --disable-warnings
echo ... Done
