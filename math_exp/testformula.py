from sympy import symbols, expand

# Define the variables
a, b, c, d = symbols('a b c d')

# Construct the expression
# We start with (a+b+c+d)**4 and subtract other combinations
# expression =(a+b+c+d)**4-(a+b+c-d)**4-(a+b-c+d)**4-(a-b+c+d)**4-(-a+b+c+d)**4+(a+b-c-d)**4+(a-b+c-d)**4+(a-b-c+d)**4
# expression = (a + b + c)**3 - (a + b - c)**3 - (a - b + c)**3 - (-a + b + c)**3 + (a - b - c)**3 + (-a + b - c)**3 - (-a - b + c)**3 - (-a - b - c)**3
expression = (a + b + c + d)**4 - (a + b + c - d)**4 - (a + b - c + d)**4 - (a - b + c + d)**4 - (-a + b + c + d)**4+ (a + b - c - d)**4 + (a - b + c - d)**4 + (a - b - c + d)**4+ (-a - b + c + d)**4 + (-a + b - c + d)**4 + (-a + b + c - d)**4- (-a - b - c + d)**4 - (-a - b + c - d)**4 - (-a + b - c - d)**4 - (a - b - c - d)**4+ (-a - b - c - d)**4
# Expand the expression to see if it simplifies to 24abcd
expanded_expression = expand(expression)

print(expanded_expression)