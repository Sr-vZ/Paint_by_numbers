#! /bin/sh
echo Starting the application
echo Dowloading dependencies
python3 -m pip install -r requirements.txt
echo Starting Flask
python3 flask_app.py