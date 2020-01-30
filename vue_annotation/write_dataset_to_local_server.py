from pymongo import MongoClient

MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017 
MONGODB_AUTHENTICATION_DB = 'annotations'
#Replace ../data/nachrichten.txt with the actual dataset
dataset = '../data/nachrichten.txt'
collection_name = "sentences"

sentences = []


with open(dataset) as f:
    for line in f:
        sentence = {}
        sentence["sent"] = line.split("\n")[0]
        sentence["annotations"] = {}
        sentences.append(sentence)



client = MongoClient(MONGODB_HOST,MONGODB_PORT)
print('authenticated on mongodb')
print('collections in Db ' + MONGODB_AUTHENTICATION_DB + ': ')
print('\t' + str(client[MONGODB_AUTHENTICATION_DB].collection_names()))
mongo_db = client[MONGODB_AUTHENTICATION_DB]
mongo_col = mongo_db[collection_name]
for item in sentences:
    mongo_col.insert_one(item)
client.close()
print('done')