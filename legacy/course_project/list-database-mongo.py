from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://admin:password123@localhost:27017/')

# List all databases
print("Available databases:")
print(client.list_database_names())

# For smart-building database, list collections
db = client['smart-building']
print("\nCollections in 'smart-building' database:")
print(db.list_collection_names())

# Show some stats
print("\nNumber of documents in demand collection:")
print(db['demand'].count_documents({}))

client.close()