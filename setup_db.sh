docker pull mongodb/mongodb-atlas-local:latest
docker run --name text2kg_mongo -p 27018:27017 -d mongodb/mongodb-atlas-local:latest
sleep 15
cd src/wikontic
python3 create_wikidata_ontology_db.py --mongo_uri "mongodb://localhost:27018/?directConnection=true"
python3 create_ontological_triplets_db.py --mongo_uri "mongodb://localhost:27018/?directConnection=true"

# to not use the ontology from wikidata, run:
# python3 create_triplets_db.py
