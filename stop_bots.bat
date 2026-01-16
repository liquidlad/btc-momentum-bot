@echo off
echo Stopping trading bots...
taskkill /FI "WINDOWTITLE eq BTC Bot*" /F 2>nul
taskkill /FI "WINDOWTITLE eq ETH Bot*" /F 2>nul
echo Bots stopped.
pause
