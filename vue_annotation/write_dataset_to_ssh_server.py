from sshtunnel import SSHTunnelForwarder
from pymongo import MongoClient

SSH_HOST = 'adrastea.ifi.uni-heidelberg.de'
LDAP_USER_NAME = 'your_ldap_user_name'
LDAP_PASSWORD = 'your_ldap_password'
MONGODB_HOST = 'localhost'
MONGODB_PORT = 'port_number_of_MongoDB (usually 27016)' 
MONGODB_AUTHENTICATION_DB = 'annotations'
MONGODB_USER_NAME = 'your_mongodb_user_name'
MONGODB_PASSWORD = 'your_mongodb_password'
#Replace ../data/nachrichten.txt with the actual dataset
dataset = '../data/nachrichten.txt'
collection_name = "sentences"

sentences = []

class sentence:
    def __init__(self,sent=""):
        self.annotations = {}
        self.sent = sent
    def __str__(self):
        return ("sentence: {{sent: {}, {{annotations: {}}}".format(self.sent, self.annotations))
    def __repr__(self):
        return self.__str__()

with open(dataset) as f:
    for line in f:
        sentences.append(sentence(line.split("\n")))



with SSHTunnelForwarder((SSH_HOST, 22), ssh_username=LDAP_USER_NAME, ssh_password=LDAP_PASSWORD,
                        remote_bind_address=('localhost', MONGODB_PORT),
                        local_bind_address=('localhost', MONGODB_PORT)) as server:
    print('connected via SSH and established port-forwarding')
    client = MongoClient(MONGODB_HOST, MONGODB_PORT)
    try:
        client[MONGODB_AUTHENTICATION_DB].authenticate(MONGODB_USER_NAME, MONGODB_PASSWORD)
        print('authenticated on mongodb')
        print('collections in Db ' + MONGODB_AUTHENTICATION_DB + ': ')
        print('\t' + str(client[MONGODB_AUTHENTICATION_DB].collection_names()))
        mongo_col = MONGODB_AUTHENTICATION_DB[collection_name]
        for item in sentences:
            mongo_col.inser_one(item)
    finally:
        client.close()
        print('done')