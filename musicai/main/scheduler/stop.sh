ps -ef | grep input.py | grep -v grep | awk '{print $2}' | xargs kill
ps -ef | grep output.py | grep -v grep | awk '{print $2}' | xargs kill
