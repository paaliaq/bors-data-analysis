import pymongo
import src.preprocess as prep


f = open("../../bors-data/src/mongodbkey.txt", "r")
mongodbkey = f.read()


client = pymongo.MongoClient(mongodbkey)
collection = client["bors-data"]["financials"]

cursor = collection.find()
all_data = list(cursor) # This doesn't scale well and hides the iteration from the programmer
processed_quarter_data = prep.prepare_quarter_data(all_data, shift_size=60)













# cursor = collection.find({}, {"Ticker": 1, "Quarterly data": 1})
# "Ticker": 1, "Name": 0, "Sector": 0, "Market": 0, "Country": 0,
#                                  "Daily data": 0, "Quarterly data": 0, "Yearly data": 0}