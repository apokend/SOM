from loguru import logger
from datetime import datetime


path = f"logs/{datetime.now().strftime('%d-%B-%Y')}/"

text = """\
<< {time:YYYY-MMMM-DD HH:mm:ss} || Enter point: {file} | Module: {name}		| {level} ||
* Called '{function}' -- '{message}'.
* Total work time:{elapsed}>>\n\n
"""

logger.add(path + "{time}.log", format=text, level="INFO")
