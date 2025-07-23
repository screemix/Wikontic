docker pull mongo
docker run --name text2kg_mongo -p 27018:27017 mongo:latest
python3 populate_db.py
python3 create_indexes.py