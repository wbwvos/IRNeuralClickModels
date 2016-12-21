Personalized Neural Click Model

Yandex personalized click log ( https://www.kaggle.com/c/yandex-personalized-web-search-challenge/data )

Sets:
	1: QD 		  query-document pairs
	2: QD+Q   	+ query prior
	3: QD+Q+D 	+ document click probabilities
	4: QD+Q+D+U + user prior

Usage:
	sets = [1,2,3,4]
	operations = [train, eval]

CLI:
	python lstm.py set operation

example train on set 1:
	python lstm.py 1 train





	 
