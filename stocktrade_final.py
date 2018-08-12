"""
stocktrader -- A Python module for virtual stock trading
TODO:  1.The stocktrader module allows the user to load historical financial data 
         and to simulate the buying and selling of shares on the stock market.
       2.This module also allows user to load the data of portfolio, 
         and evaluate the portfolio on the chosen day.
       3.The two trading strategies go through all trading days in the dictionary `stocks`, 
         buy and sell shares automatically,based on specific rules.
        
Also fill out the personal fields below.

Full name: Yifan Yu
StudentId: 9959198
Email: yifan.yu-2@student.manchester.ac.uk
"""
from datetime import datetime
import re
import csv
from pprint import pprint
import os
from os.path import isfile, join
import math
# modules for task 10
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.tsa.stattools as st

class TransactionError(Exception):
    pass

class DateError(Exception):
    pass

class LinAlgError(Exception):
    pass

stocks = {}
portfolio = {}
transactions = []


# Task 1
def normaliseDate(s): 
    """
    input a string `s`, and return a data string of the form 'YYYY-MM-DD'
    inputs are accepted in three formats: YYYY-MM-DD,YYYY/MM/DD and DD.MM.YYYY
    """
    
    # the following 4 lines follow a similar code on 
    # https://stackoverflow.com/questions/33051147/normalize-different-date-data-into-single-format-in-python as retrieved on 30/03/2018
    # format: DD.MM.YYYY  
    if re.search(r'^\d{1,2}\.\d{1,2}\.\d{4}$',s):   
        date_str = datetime.strptime(s,'%d.%m.%Y')
    # format: YYYY-MM-DD 
    elif re.search(r'^\d{4}\-\d{1,2}\-\d{1,2}$',s):
        date_str = datetime.strptime(s,'%Y-%m-%d')
    # format:  YYYY/MM/DD
    elif re.search(r'^\d{4}\/\d{1,2}\/\d{1,2}$',s): 
        date_str = datetime.strptime(s,'%Y/%m/%d')   
    else:        
        raise DateError("Date Format is not allowed:{0}".format(s))
    return date_str.strftime("%Y-%m-%d")


# Task 2
def loadStock(symbol):
    """
    input a string `symbol`,
    load corresponding stock data from csv file into dictionary `stocks`
    """
    symbol_dict = {}
    # convert the symbol to uppercase
    symbol = symbol.upper()
    symbol_file = '{}.csv'.format(symbol)
    path = os.path.join('stockdata',symbol_file)
    if os.path.exists(path) is False:
        raise FileNotFoundError("The corresponding company data is not contained in stockdata")
    else:    
        with open (path, mode="rt", encoding="utf8") as ifile:
            # skip the first line 
            ifile.readline()
            data_reader = csv.reader(ifile, delimiter=",")
            for row in data_reader:
                Date = normaliseDate(row[0])
                for i in range(1,5):                    
                    # raise ValueError if a line is of invalid format
                    try:
                        row[i] = float(row[i])                        
                    except ValueError:
                        raise ValueError("Some lines in the CSV file is of invalid format")
                symbol_dict[Date] = row[1:5]    
        stocks[symbol] = symbol_dict        
        return 


# Task 3
def loadPortfolio(fname='portfolio.csv'):
    """
    input a string `fname`
    load the data from file and assign to `portfolio` dictionary
    and load corresponding stock data into `stocks` dictionary
    """
    # ensure portfolio and transactions are emptied before loading new data 
    portfolio.clear() 
    transactions.clear()
    
    # check if the file exists, otherwise raise an exception
    if os.path.exists(fname) is False:
        raise FileNotFoundError("The file is not found")
    else:
        with open(fname, mode="rt",encoding="utf8") as f:
            f_reader = csv.reader(f,delimiter= ",")
             # the following 3 lines follow a similar code on 
             # https://stackoverflow.com/questions/2081836/reading-specific-lines-only-python as retrieved on 01/04/2018
            for i,line in enumerate(f_reader):  
                try: 
                    if i == 0:
                        portfolio["date"] = normaliseDate(line[0])
                    if i == 1:
                        portfolio["cash"] = float(line[0])
                        # check if cash is nonnegative 
                        if portfolio["cash"] < 0:
                            raise ValueError("cash cannot be a negative floating point number")                   
                    if i >= 2:
                        portfolio[line[0]] = int(line[1])
                        loadStock(line[0])
                except ValueError:
                    raise ValueError("The format of a line in the file is invalid")
    return 


# Task 4
def valuatePortfolio(date=None,verbose=False):
    """
    input `date`(the date of portfolio by default) and `verbose`(Boolean Value, True by default)
    return the total value(floating point number) and also a table if `verbose`=True
    """
    if date is None:
        date = portfolio.get('date')
    date = normaliseDate(date)
    total_stock = list() # list of all stocks' information in portfolio
    # add Cash information to `total_stock` first
    total_stock.append({'Capital type': 'Cash', 'Volume': 1, 'Val/Unit*': portfolio.get('cash'), 'Value in £*': portfolio.get('cash')})
    total_value = portfolio.get('cash') # add cash to total value first 
    
    #compare `date` and the date: 
    # it is unallowed if `date` is earlier than the date of the portfolio
    if date < portfolio.get('date'): 
        raise DateError("The date is earlier than the date of the portfolio")
    else:            
        for stock in portfolio.keys():        
            # find the stocks in portfolio by checking it is also a key in `stocks`
            if stock in stocks.keys(): 
                # all information of a specific stock in portfolio
                stock_value = dict()
                # Capital type for each stock in portfolio
                stock_value["Capital type"] = "Shares of {}".format(stock)                
                volume = portfolio.get(stock)# volumn for each stock in portfolio
                stock_value["Volume"] = volume          
                if date not in stocks[stock].keys(): # check if the date is a trading day
                    raise DateError("The date is not a trading day")
                else:    
                     #value/unit for each stock in portfolio
                     #found in dict `stocks` with key=stock and then key=date, the lowest price has index 2                    
                    unit_value = stocks[stock][date][2]
                    stock_value["Val/Unit*"] = unit_value 
                    value = volume * unit_value
                    stock_value["Value in £*"] = value
                    
                    total_value += value
                    total_stock.append(stock_value)
        
        
        if verbose == True:
            print("Your portfolio on {}:".format(date))
            print("[* share values based on the lowest price on {}]\n".format(date))
            print("{0:<22} | {1} | {2} | {3:^8}".format("Capital type","Volume","Val/Unit*","Value in £*"))
            print("-" * 23 + "+" + "-" * 8 + "+" + "-" * 11+ "+" + "-" * 13)
            for items in total_stock:
                print("{Capital type:<22} | {Volume:>6} | {Val/Unit*:9.2f} | {Value in £*: 11.2f}".format(**items))
            print("-" * 23 + "+" + "-" * 8 + "+" + "-" * 11+ "+" + "-" * 13)
            print("TOTAL VALUE{:>46.2f}".format(total_value))
        return total_value
    







# Task 5    
def addTransaction(trans,verbose=False):
    """
    input dictionary `trans` and Boolean variable `verbose`(False by default)
    update dictionary `portfolio`, and print performed transaction if verbose = True
    """
    date = normaliseDate(trans.get('date')) # obtain information in `trans`
    symbol = trans.get('symbol')
    volume = trans.get('volume')
    
    if symbol not in stocks.keys():
            raise ValueError("The symbol in transaction is not in stocks dictionary")
    if date < portfolio['date']:
        raise DateError("The date of transaction is earlier than that of portfolio")
    # get the lowest price when selling, and the highest price when buying
    price = stocks[symbol][date][2 if volume < 0 else 1]    
    total_price = price * volume 
    cash = portfolio.get('cash') - total_price   
    
    if (cash < 0 or 
       portfolio.get(symbol) is None and volume < 0 or 
       portfolio.get(symbol) is not None and portfolio.get(symbol) + volume < 0): 
        raise TransactionError("Not enough cash or Not enough shares to sell")        
    # Update cash, shares, date in portfolio
    portfolio['date'] = date 
    portfolio['cash'] = cash    
    if portfolio.get(symbol) is not None:
        if portfolio.get(symbol) + volume !=0:
            portfolio[symbol] = portfolio.get(symbol) + volume
        else:
            del portfolio[symbol]
    else:
        portfolio[symbol] = volume
    #append `trans` to list `transactions`
    transactions.append(trans) 
    
    
    if verbose == True:   
        # the following 2 lines follow a similar code on 
        # https://stackoverflow.com/questions/10135080/is-there-a-way-to-add-a-conditional-string-in-pythons-advance-string-formatting as retrieved on 02/04/2018
        args = [date, 'Sold' if volume<0 else 'Bought',abs(volume), symbol, abs(total_price),'Available' if volume<0 else 'Remaining',cash]
        trans_info = "{}: {} {} shares of {} for a total of {:.2f} \n{} cash: £ {:.2f}".format(*args)
        print(trans_info)    
    return 


# Task 6
def savePortfolio(fname="portfolio.csv"):
    """
    input string `fname`(including ".csv")
    save the updated dictionary `portfolio` to csv, named `fname` 
    """   
    # the following 4 lines follow a similar code on 
    # https://stackoverflow.com/questions/8685809/python-writing-a-dictionary-to-a-csv-file-with-one-line-for-every-key-value as retrieved on 02/04/2018   
    with open (fname, mode="wt",encoding="utf8") as csv_file:
        f_writer = csv.writer(csv_file)
        for key, value in portfolio.items():
            f_writer.writerow([key, value])


# Task 7   
def sellAll(date=None, verbose=False):
    """
    input string `date` and Boolean variable `verbose`(False by default)
    sell all shares in portfolio on a particulat date
    """
    if date is None:
        date = portfolio.get('date')
    date = normaliseDate(date)
    for key in portfolio.copy(): # filter the key of symbols 
        if key in stocks.keys():            
            key_trans = {'date' : date, 'symbol' : key, 'volume' :-portfolio[key]}
            addTransaction(key_trans,verbose)
    return 

# Task 8
def loadAllStocks():
    """
    load historic stock data from subdirectory into dictionary `stocks`
    """
    # the following 1 lines follow a similar code on 
    # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory as retrieved on 02/04/2018      
    # and use regular expression to delete hidden file such as '.DS_Store'
    all_file = [file for file in os.listdir('stockdata')if isfile(join('stockdata',file)) and re.search('\.csv',file)] 
    for x in all_file:
        try: # ignore the file if loading fails
            x = re.sub('\.csv$','',x) # delete substring '.csv'
            loadStock(x)
        except ValueError:
            pass
    return 


# Task 9
def H(s,j):
    """
    input the stock `s` and `j`th trading day
    return the high price on that day
    """    
    # the following 1 lines follow a similar code on 
    # https://stackoverflow.com/questions/3097866/access-an-arbitrary-element-in-a-dictionary-in-python as retrieved on 03/04/2018         
    # get the first dictionary inside dictionary `stocks` 
    first_dict = next (iter (stocks.values())) 
    # the list of all trading days from `stocks` dictionary
    trading_day = sorted(list(first_dict.keys()))    
    trade_date = trading_day[j]
    high_price = stocks.get(s).get(trade_date)[1]   
     
    return high_price

def Q_buy(s,j):
    """
    input the stock `s` and `j`th trading day
    return the ratio of the high price on that day with the average high price of all previous ten days
    """    
    if j >= 9:
        denominator = 0
        for i in range(0,10):
            denominator += H(s,j-i) # get the denominator by summation            
        answer = 10 * H(s,j)/denominator
    else:
        answer = 0
    return answer
    
def L(s,j):    
    """
    input the stock `s` and `j`th trading day
    return the low price on that day
    """      
    first_dict = next (iter (stocks.values())) # get the first dictionary inside dictionary `stocks`
    trading_day = sorted(list(first_dict.keys()))
    trade_date = trading_day[j]
    low_price = stocks.get(s).get(trade_date)[2]      
    return low_price





def tradeStrategy1(verbose=True):
    """
    Go through all trading days in the dictionary `stocks` 
    Buys and sells shares automatically   
    The trade strategy is 'buy as much as possible & sell all' 
    And only consider buying new shares on the next trading day
    """   
    # the following 1 lines follow a similar code on 
    # https://stackoverflow.com/questions/3097866/access-an-arbitrary-element-in-a-dictionary-in-python as retrieved on 03/04/2018           
    first_dict = next (iter (stocks.values())) # get the first dictionary inside dictionary `stocks`
    trading_day = sorted(list(first_dict.keys()))
    begin_date = portfolio.get('date')
    cash = portfolio.get('cash')
    
    if begin_date in trading_day:
        j = trading_day.index(begin_date) # get the index of date of portfolio if it is trading day
    else: # get the next trading day 
        closest_date = min([x for x in trading_day if x > begin_date])
        if trading_day.index(closest_date) > 9: # start buying share on the next trading day
            j = trading_day.index(closest_date)
        else: #buy shares on 2012-01-16 if the index of next trading day is still less than 10
            j = 9

               
    while j < len(trading_day):    
        cash = portfolio.get('cash')
        # the following 1 lines follow a similar code on 
        # https://stackoverflow.com/questions/37693373/sort-python-list-with-two-keys-but-only-one-in-reverse-order as retrieved on 08/05/2018           
        # choose the stock with maximal quotient `Q_buy(s,j)`, also by lexicographical order
        sym_list = sorted(stocks.keys(), key= lambda s:(-Q_buy(s,j),str.upper))
        sym = sym_list[0]
        
        unit_value = stocks[sym][trading_day[j]][1]
        vol = math.floor(cash/unit_value)
        trans = {'date': trading_day[j],'symbol': sym, 'volume': vol} # buy the largest possible volume
       
        addTransaction(trans,True) 
           
        k = j + 1
        while k< len(trading_day):            
            # Q_sell(k)<0.7 --> already lost at least 30%
            # Q_sell(k)>1.3 --> make a profit of at least 30%
            if L(sym,k)/ H(sym,j) > 1.3 or L(sym,k)/ H(sym,j) < 0.7: 
                sell_trans = {'date':trading_day[k],'symbol':sym,'volume':-vol}
                addTransaction(sell_trans,True)
                break
            k += 1 
        j = k + 1 # only consider buying new shares on the following trading day
        
    return 
    



# Task 10
def trading_day_index(date):
    """
    input a string `date`
    return its index in trading day list
    """
    date = normaliseDate(date)
    first_dict = next (iter (stocks.values())) # get the first dictionary inside dictionary `stocks`
    trading_day = sorted(list(first_dict.keys()))
    
    return trading_day.index(date)


def predict_stock(symbol,start_date,end_date,predict_duration,buy=True,verbose=True):
    """
    Input strings `symbol` ,`start_date`,`end_date` and `predict_duration`
    and Boolean value `buy` and `verbose`, print graphs and explanation if verbose is True
    plot the corresponding stock prices
    raise FileNotFoundError if it is not in stockdata folder
    """
    symbol = symbol.upper()
    start_date = normaliseDate(start_date)
    end_date = normaliseDate(end_date)
    i = trading_day_index(end_date)
    
    price_stocks = dict()
    try:
        #create a new dictionary of `symbol` with `start_date` and `end_date`
        for day in stocks[symbol].keys():
            if day >= start_date and day <= end_date:
                if buy == True: # buy stocks at high price
                    price_stocks[day] = stocks[symbol][day][1]
                else: # sell stocks at low price
                    price_stocks[day] = stocks[symbol][day][2]
        if verbose == True:
            plt.plot(*zip(*sorted(price_stocks.items())))
            plt.title("The high stock price of {} during {} and {}".format(symbol,start_date,end_date))
            plt.show()
                
        # the following 1 lines follow a similar code on 
        # https://stackoverflow.com/questions/17839973/construct-pandas-dataframe-from-values-in-variables as retrieved on 04/14/2018                         
        df= pd.DataFrame(price_stocks,index=[0])
        df.index = pd.to_datetime(df.index)
        # the following 1 lines follow a similar code on 
        # https://stackoverflow.com/questions/33246771/convert-pandas-data-frame-to-series as retrieved on 04/14/2018                             
        # convert pandas data frame to series
        ts = df.iloc[0]
        ts.head().index
        ts_log = np.log(ts)
        log_price_matrix = ts_log.as_matrix()
        if verbose == True:
            plt.plot(*zip(*sorted(ts_log.items())))
            plt.title("The high stock price of {} after log transformation during {} and {}".format(symbol,start_date,end_date))
            plt.show()
            
        # the following 4 lines follow a similar code on 
        # https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/ as retrieved on 04/14/2018                                
        
        model = ARIMA(log_price_matrix,order=(1,1,0))
        model_fit = model.fit(disp=0)              
        predictions = model_fit.predict(i+1,i+predict_duration,typ='levels')
        actual_predictions = np.exp(predictions)
        
                
        if verbose == True:
            print(model_fit.summary()) 
            plt.plot(actual_predictions)
            plt.title("The prection of high stock price of {} in next {} days".format(symbol,predict_duration))
            plt.show()
            
        
    except KeyError:
        raise FileNotFoundError("The corresponding company data is not contained in stockdata")
    return actual_predictions


def price_increase(symbol,date,verbose=False):
    """
    input strings `symbol` and `date`
    and Boolean Value `verbose`
    check if its stock buying price is predicted to increase in the following days 
    """
    symbol = symbol.upper()
    loadStock(symbol)
    first_dict = next (iter (stocks.values())) # get the first dictionary inside dictionary `stocks`
    trading_day = sorted(list(first_dict.keys()))
    if trading_day_index(date) > 20:
        start_date=trading_day[trading_day_index(date)-15]
    else:
        start_date=trading_day[10]
    
    actual_predictions = predict_stock(symbol,start_date,end_date=date,predict_duration=5,buy=True,verbose=False)
    # the following 2 lines follow a similar code on 
    # https://stackoverflow.com/questions/30734258/efficiently-check-if-numpy-ndarray-values-are-strictly-increasing as retrieved on 04/14/2018                                 
    if all(x <= y for x, y in zip(actual_predictions, actual_predictions[1:])):
        return True
    else:
        return False
    
def buy_stock(symbol,j):
    """
    input stock name `symbol` and index of current date
    return the predicted increase of stock price
    """
    
    symbol = symbol.upper()
    loadStock(symbol)
    first_dict = next (iter (stocks.values())) # get the first dictionary inside dictionary `stocks`
    trading_day = sorted(list(first_dict.keys()))
    date = trading_day[j]
    if trading_day_index(date) > 20:
        start_date=trading_day[trading_day_index(date)-15]
    else:
        start_date=trading_day[10]
    date = trading_day[j]
    actual_predictions = predict_stock(symbol,start_date,end_date=date,predict_duration=5,buy=True,verbose=False)
    
    
    increase = float(actual_predictions[4])-float(actual_predictions[0])
  
    return increase





def price_decrease(symbol,date,verbose=False):
    """
    input strings `symbol` and `date`
    and Boolean Value `verbose`
    check if its stock selling price is predicted to decrease in the following days 
    """
    symbol = symbol.upper()
    loadStock(symbol)
    first_dict = next (iter (stocks.values())) # get the first dictionary inside dictionary `stocks`
    trading_day = sorted(list(first_dict.keys()))
    if trading_day_index(date) > 20:
        start_date=trading_day[trading_day_index(date)-15]
    else:
        start_date=trading_day[10]
    
    actual_predictions = predict_stock(symbol,start_date,end_date=date,predict_duration=5,buy=False,verbose=False)
    # the following 2 lines follow a similar code on 
    # https://stackoverflow.com/questions/30734258/efficiently-check-if-numpy-ndarray-values-are-strictly-increasing as retrieved on 04/14/2018                                 
    if all(x >= y for x, y in zip(actual_predictions, actual_predictions[1:])):
        return True
    else:
        return False
    
    
def sell_stock(symbol,j):
    """
    input stock name `symbol` and index of current date
    return the predicted decrease of stock price
    """
    
    symbol = symbol.upper()
    loadStock(symbol)
    first_dict = next (iter (stocks.values())) # get the first dictionary inside dictionary `stocks`
    trading_day = sorted(list(first_dict.keys()))
    date = trading_day[j]
    if trading_day_index(date) > 20:
        start_date=trading_day[trading_day_index(date)-15]
    else:
        start_date=trading_day[10]
    date = trading_day[j]
    actual_predictions = predict_stock(symbol,start_date,end_date=date,predict_duration=5,buy=False,verbose=False)
    
    
    decrease_rate = float(actual_predictions[4])/float(actual_predictions[0])
    if decrease_rate > 1.1 or decrease_rate < 0.9:
        return True
    else:
        return False

def tradeStrategy2(verbose=True):
    # get the first dictionary inside dictionary `stocks`
    first_dict = next (iter (stocks.values())) 
    trading_day = sorted(list(first_dict.keys()))
    begin_date = portfolio.get('date')
    cash = portfolio.get('cash')
    if begin_date in trading_day:
        # get the index of date of portfolio if it is trading day
        j = trading_day.index(begin_date) 
    else: # get the index of the next trading day
        closest_date = min([x for x in trading_day if x > begin_date])
        if trading_day.index(closest_date) < 15:
            j = 20
        else:
            j = trading_day.index(closest_date) 
   
    
    
    
    
    while j < len(trading_day):    
        cash = portfolio.get('cash')
        # find the stock which is prediced to increase daily in the following five days
        increase_stock_list = list()
        for stock in stocks.keys():
            if price_increase(stock,date=trading_day[j],verbose=False) is True:
                increase_stock_list.append(stock)
        # find the stock with predicted largest increase 
        buy_stock_list = sorted(increase_stock_list, key= lambda s:-buy_stock(s,j))        
        
        # buy stocks at the next trading day if all stock pricez are predicted to decrease on that day            
        if len(buy_stock_list) == 0:
            pass
        else: 
            sym = buy_stock_list[0]
        
            unit_value = stocks[sym][trading_day[j]][1]
            vol = math.floor(cash/unit_value)
            trans = {'date': trading_day[j],'symbol': sym, 'volume': vol} # buy the largest possible volume
           
            addTransaction(trans,True) 
            # only sell shares of stocks after keeping it for 20 days 
            k = j + 1
            while k< len(trading_day):            
                
                if price_decrease(sym,date=trading_day[k],verbose=False) is True: 
                    sell_trans = {'date':trading_day[k],'symbol':sym,'volume':-vol}
                    addTransaction(sell_trans,True)
                    break
                k += 1 
            j = k + 1 # only consider buying new shares on the following trading day
            

    return 



 
    
    
def main():    
    """
    s = '8.5.2012'
    print(normaliseDate(s))        
    symbol = "ezj"    
    pprint(loadStock(symbol))
    print(loadPortfolio())        
    print(valuatePortfolio('2012-2-6', True))
    print(addTransaction({ 'date':'2013-08-12', 'symbol':'SKY', 'volume':-5 }, True))        
    fname = "portfolio5.csv"
    savePortfolio(fname)
    sellAll(verbose=True)
    loadAllStocks()
     
    print(H('BATS',1))
        
    
    print(L('SKY',1))
    print( Q_buy('SKY',0))
    """
    """
    loadPortfolio('portfolio0.csv')    
    loadAllStocks()
    #valuatePortfolio(verbose=True)
    tradeStrategy1(verbose=True)
    valuatePortfolio(date="2018-03-13", verbose=True)
    """
    
    
    loadPortfolio('portfolio0.csv')    
    loadAllStocks()
    valuatePortfolio(date='2012-08-06',verbose=True)
    predict_stock(symbol='GFS',start_date='2012-03-13',end_date='2013-03-25',predict_duration=5,buy=False,verbose=True)
    #print(price_increase(symbol='BATS',date='2013-01-09',verbose=True))
    #print(buy_stock(symbol='BATS',date='2013-01-09'))
    tradeStrategy2(verbose=True)
    #valuatePortfolio(date="2018-03-13", verbose=True)
    #print(predice_increase('VOD',5))
    
main()
    
    



"""
# the following allows your module to be run as a program
if __name__ == '__main__' or __name__ == 'builtins':
    main()
"""