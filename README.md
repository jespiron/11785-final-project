# About

Final project for 11785.

**Group name:** TODO

**Group Members**
* [Jessica Ruan (jwruan)](https://github.com/jespiron)
* [Brandon Dong (bjdong)](https://github.com/sad-ish-cat)
* [Wonsik Shin (wonsiks)](https://github.com/ceteris11)
* [Aradhya Talan (atalan)](https://github.com/aradhyatalan)

**[TODO] Project Summary:** Explain project. This is a great project, very great, very bigly, deep learning is deep, databases are based

Cardinality estimation for DBMSes, [high-level motivation](https://drive.google.com/file/d/17HtE_3dq_qvoLfBEnw9QUmRuJgxaqqkx/view?usp=sharing)

[TODO] replace with proposal

# Testing

We will use CMU optd's cardinality benchmarking feature to test performance of the cost model. [optd](https://github.com/cmu-db/optd/tree/main) has been included as a Github submodule in this repository.

Before running the benchmark, you will need to manually run Postgres on your machine. Below I included instructions for doing so, let me ([@jespiron](https://github.com/jespiron)) know if it doesn't work.

## Setup
1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Pull the postgres image

```docker pull postgres```

3. Start the postgres server

```docker run -p 5432:5432 --name mypostgres -e POSTGRES_USER=your-username -e POSTGRES_PASSWORD=your-password -e POSTGRES_INITDB_ARGS=--auth-host=scram-sha-256 -d postgres```

4. Confirm postgres server is up

```curl localhost:5432```

If you see `"curl: (52) Empty reply from server` it works

## Running the Benchmark
```
 cargo run --release --bin optd-perfbench cardbench tpch --scale-factor 0.01 --pguser your-username --pgpassword your-password
 ```

# Contributing

Please name branches `your-nickname/name-of-change`. Isn't strict as long as we know who's who

TODO set up directory for models and explain dirtree
