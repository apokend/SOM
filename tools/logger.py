#---------------------------+
#        Version:  1.01     +
#   Status: Ready to Prod   +
#   Author: Shevchenko A.A. +
#-------------------------- +

from loguru import logger
from datetime import datetime

# Path to logs
path = f"logs/{datetime.now().strftime('%d-%B-%Y')}/"

# Our log text
text = """\
<< {time:YYYY-MMMM-DD HH:mm:ss} || Enter point: {file} | Module: {name}		| {level} ||
* Called '{function}' -- '{message}'.
* Total work time:{elapsed}>>\n\n
"""

# Create logger
logger.add(path + "{time}.log", format=text, level="INFO")
