"""
This is just some testing code for the stocktrader module.
It corresponds to the code given in Task 5.5.
Feel free to modify it and use it for testing exceptional cases. 
"""

import stocktrade_final as s

s.loadPortfolio()
val1 = s.valuatePortfolio(verbose=True)
trans = {'date':'2013-08-12', 'symbol':'SKY', 'volume':-5 }
s.addTransaction(trans,verbose=True)
val2 = s.valuatePortfolio(verbose=True)
print("Hurray, we increased our portfolio value by £{:.2f}!\n".format(val2-val1))


print('-'*60)
s.loadPortfolio('portfolio0.csv')
s.loadAllStocks()
s.valuatePortfolio(verbose=True)
s.tradeStrategy1(verbose=True)
s.valuatePortfolio('2018-03-13',verbose=True)
