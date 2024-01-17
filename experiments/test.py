import re

from decimal import Decimal, getcontext

# Set the precision (number of significant digits)
getcontext().prec = 300
question = '-210907186319801 + -0.02'
print(question)
operands = re.findall(r'-?\d+(?:\.\d+)?', question)
print(operands)
print(Decimal(operands[0])+Decimal(operands[1])-Decimal(-210907186319801.02))