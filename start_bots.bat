@echo off
echo Starting BTC and ETH trading bots...
cd /d C:\Users\eliha\btc-momentum-bot

start "BTC Bot" cmd /k python runbot.py --exchange paradex --ticker BTC
timeout /t 5
start "ETH Bot" cmd /k python runbot.py --exchange paradex --ticker ETH

echo Bots started in separate windows.
echo Close the windows to stop the bots.
