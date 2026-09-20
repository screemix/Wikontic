docker pull mongodb/mongodb-atlas-local:latest
# mongod inside this image always listens on 27017 internally, regardless
# of the published host port -- map to that, not to 27018:27018.
docker run --name text2kg_mongo -d -p 27018:27017 mongodb/mongodb-atlas-local:latest
python -m wikontic.create_wikidata_ontology_db --backend mongodb --mongo_uri "mongodb://localhost:27018/?directConnection=true" --database wikidata_ontology
python -m wikontic.create_ontological_triplets_db --backend mongodb --mongo_uri "mongodb://localhost:27018/?directConnection=true" --db_name triplets_db

# to not use the ontology from wikidata, run:
# python -m wikontic.create_triplets_db --backend mongodb --mongo_uri "mongodb://localhost:27018/?directConnection=true" --db_name triplets_db
