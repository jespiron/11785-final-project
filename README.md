# About

Final project for 11785.

Group name: TODO

Group members: add your 

TODO: Explain project. This is a great project, very great, very bigly, deep learning is deep, databases are based

# Testing

We will use CMU optd's cardinality benchmarking feature to test performance of the cost model. [optd](https://github.com/cmu-db/optd/tree/main) has been included as a Github submodule in this repository.

Before running this, you will need to manually run Postgres on your machine. Below I included instructions for doing so, let me (@jespiron) know if it doesn't work.

## Setup
1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Pull the postgres image: `docker pull postgres`
3. Start the postgres server
```docker run -p 5432:5432 --name mypostgres -e POSTGRES_USER=your-username -e POSTGRES_PASSWORD=your-password -e POSTGRES_INITDB_ARGS=--auth-host=scram-sha-256 -d postgres```
4. Confirm postgres server is up
```curl localhost:5432 # Expected "curl: (52) Empty reply from server"```

## Running the Benchmark
```
 cargo run --release --bin optd-perfbench cardbench tpch --scale-factor 0.01 --pguser jruan --pgpassword mysecretpassword
 ```