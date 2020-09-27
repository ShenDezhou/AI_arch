from handcalcs.decorator import handcalc

@handcalc()
def function(n, B):
    y = n**2.8


latex, _ = function(1,2)
print(latex)
