import pymongo
import src.preprocess as prep


f = open("../bors-data/src/mongodbkey.txt", "r")
mongodbkey = f.read()


client = pymongo.MongoClient(mongodbkey)
collection = client["bors-data"]["financials"]

cursor = collection.find()
all_data = list(cursor)  # This doesn't scale well and hides the iteration from the programmer

processed_quarter_data, missing_count = prep.fill_quarter_data(all_data, shift_size=60)
# TODO Clean moving average of index
# TODO Fill with yearly data

ml_ready_set = prep.QuarterDataMLReady(processed_quarter_data, "reported_mean_Close",
                                       ["reported_quarter", "report_End_Date"])
# TODO decide which date to use


