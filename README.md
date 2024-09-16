# About

Final project for 11785.

**Group name:** Group 22

**Group Members**
* [Jessica Ruan (jwruan)](https://github.com/jespiron)
* [Brandon Dong (bjdong)](https://github.com/sad-ish-cat)
* [Wonsik Shin (wonsiks)](https://github.com/ceteris11)
* [Aradhya Talan (atalan)](https://github.com/aradhyatalan)

**[TODO] Project Summary:** Explain project. This is a great project, very great, very bigly, deep learning is deep, databases are based

Cardinality estimation for DBMSes, [high-level motivation](https://drive.google.com/file/d/17HtE_3dq_qvoLfBEnw9QUmRuJgxaqqkx/view?usp=sharing)

[TODO] replace with proposal

# Contributing

Please name branches `your-nickname/name-of-change`. Isn't strict as long as we know who's who

When your PR is approved, please select **[Squash and merge](https://www.lloydatkinson.net/posts/2022/should-you-squash-merge-or-merge-commit/)** from the dropdown. This leads to a much cleaner commit history!

After merging the PR, clean up your branch locally and remotely.
1. Locally: `git branch -D your-nickname/name-of-change`
2. Remotely: deleted automatically since "Automatically delete head branches" is enabled. If it doesn't work, can delete manually by clicking the trash icon next to your branch.

**Where to open PR from?**

If you're making a change that's not related to model training, can make a branch on this repo and open a PR.

If you're contributing a model, please fork this repo and open a PR from your fork.

**Set up:**
1. Fork this repository
2. In your fork, add this repo as a remote `git remote add upstream https://github.com/jespiron/11785-final-project.git`
3. Pull changes with `git fetch upstream -p`. The `-p` prunes any branches that were deleted upstream

Your forked repo will hold models that you're experimenting with and not ready to share with the group. Since I anticipate we'll be experimenting with a lot of models, it'll be cleaner if we isolate models on the repo level, rather than a branch per model or *shudder* a branch with mounds of models.

The reason for this is the more branches we have on remote, the more branches other contributors have to scroll through.

**Workflow:**
1. Train models (see README under `/models`)
2. Test the models with `optd` (see Testing section for how)
3. Open PR

TODO integrate model output with optd's benchmarking

# Testing

We will use CMU optd's cardinality benchmarking feature to test performance of the cost model. [optd](https://github.com/cmu-db/optd/tree/main) has been included as a Github submodule in this repository.

Before running the benchmark, you will need to manually run Postgres on your machine. Below I included instructions for doing so via Docker container, let me ([@jespiron](https://github.com/jespiron)) know if it doesn't work. An alternative way is in the `patrick/` folder as suggested in [docs](https://cmu-db.github.io/optd/cost_model_benchmarking.html).

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

See [docs](https://cmu-db.github.io/optd/cost_model_benchmarking.html) for more on the benchmarking tool
