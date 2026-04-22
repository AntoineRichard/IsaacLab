
# Multi backend large scale evaluation system for Isaaclab

Highlevel requirements:
- must run a large set of trainings with different backendsand report their training metrics. 
- must provide a dashboard for easy visualization
- can reuse / upgrade the benchmarking scripts.
- Interest in both runtime perf, start-up perf, system resources consumption, and reward reached. 
- Ideally we should transition to a metric based evaluation system, but this can wait. 

For all the following tasks, use the brainstorming skill to scope the work that's required. Don't blindly go and implement these tasks. This harness would not live in the isaaclab repo directly, but to make it happen we may need to modify the main isaaclab repository. It's important to properly separate the work. For now, we'll backe all our work in the repo to make development easy but we need to consider the separation. As such we should not push anything, we can of course commit things locally in a new branch, but we should not push to a remote. 

## T0: Naming

Based on the info in this doc let's find a cool name, maybe from a movies?

**Decided: Odin** (Norse mythology). Odin is the all-seer who receives information from his two ravens, Hugin (thought) and Munin (memory), who fly out over the world and report back. This maps tightly onto a distributed benchmark harness: a controller that dispatches runners, receives their results, and aggregates everything into one view. The pantheon gives us a coherent naming vocabulary for later subsystems:

- **Odin** — controller (dispatch + aggregation)
- **Hugin / Munin** — benchmark runners (one per learning framework, e.g. RSL-RL and SKRL)
- **Valhalla** — results archive / dashboard (T4)
- **Asgard** — the compute cluster (T3)
- **Valkyries** — worker nodes that carry results back (T3)
- **Bifrost** — inter-node communication / SSH transport (T3)
- **Yggdrasil** — the IL2.3.x ↔ IL3.x bridge (T5)

Feature branch: `antoiner/feat/odin` (off `develop`).

## T1: The evaluation

We need to run benchmarked trainings, with different seeds, and collect the results. The naming of these trainings should be transparent i.e. if training ant, let's not have a hash or just ant. But maybe BACKEND\_TASK\_DATE\_SEED. Or someting like that. Setting this up should require minimum work, as almost all the elements should be there. Our main interests lie in runtime perf, start-up perf, system resources consumption, and reward reached.

We could run a dry run on something like ant direct, to create a benchmark file that other tasks could look up.

We'll need to check how the reward / episode length are reported. These are key indicators of the training quality, and if we just rely on the last measured numbers we could end-up with biased results. A potential approach to solve this is to add a EMA metric in the benchmark with a small alpha so that it smoothes over the last ~20 measurements. This is key in providing accurate readings. Same things could be done about the resource measurements, but they may naturally come with mean + std, which is largely sufficient.

commit and version logging: need to log kit version, isaacsim, newton, warp, mjwarp. Need to log GPU type and CPU type. Let's make sure it's all accessible. If not we'll need to add it. Note, we have a dedicated start-up script with dense profiling capabilities that might be interesting to leverage.

## T2.1: Building the list of environments to train

We need to know what we can/should run. Our learning framework of preference would be rsl\_rl except for vision workflows where we would switch to skrl. We need to explore what environments PhysX provide and see what we want to run. We could start by making a list that you would present to me, and I'll send back a filtered list. Then we'll do the same with newton. But for newton, we'll also look at what it is not supporting from the filtered list. We'll need to identify if something is missing to enable these environments, and if not consider adding them. With newton, we still have a couple of gaps in the API notably: SDF collisions required for rough terrain locomotion and fancy assembly tasks like nut and bolt. Also missing tendons support. Manipulation examples are not well tested but could be explored. At the end we'll have two lists and a good idea of the gaps for newton. So 3 deliverables 2 lists of environments, and a doc summarizing the API Gaps in newton. We're leaving ovphysx backend out for now.

## T2.2: Dense startup profiles

We should explore the dense profiling capabilities in Isaaclab to see what should be reported.

## T3: Distributed running system. 

Considering the breadth of tasks that need to be tested, we may need to consider a distributed framework where a controller node dispatches work to different machines. NVIDIA provides me with resources required to do this. My idea is to instantiate a set of N machines, I'll provide a list of IPs, and these machines will be accessible through SSH. They will all share the same hardware specification. The first thing the controller will have to do is pull the isaaclab repository, and run the docker setup. (./docker/container.py start) Then wait for docker to start. Once the docker is started, the controller can start to dispatch benchmark training jobs to it. Ideally we want to be able to monitor these jobs, see if they are progressing, something crashed and so on... Typically, the terminal dump from these scripts is pretty verbosy so we should be able to leverage that directly. Collecting the PID of the launched job with a monitor attached to it could also be used to notify the controller when jobs are done or died. We may want to have a simple local web interfaces that shows the task list, what's left, and which node is working on what.
When a job is done the monitor should pull the resutls.

I don't know if an AI agent is required to monitor any of this, but I feel like a lot of this could be automated without requiring agents.
The code will ingest the lists done in the previous task.

## T4: Reporting system & aggregation

After a complete run, we want to report all the failures experienced. This is key to help us catch complete failures early and debug our code. Aggregating all the results is also important, if we chose to do multi-seed runs (to test the stability / reliability of the training) we need to be able to report aggregated measurements or flag radically different runs. Similarly when reporting reward numbers, we should never report the bare reward, but use the EMA / averaged variations this is key to providing reliable results. Ideally we want to compound all the individual runners' results into a single file, while keeping the original results for safe keeping. If a runner failed completely, i.e. no results, this should also be ahanlded so we know if failed. 

Dashboarding. This is extremely important. I would recommend using dash and plotly for this. The goal is two folds, 1: provide high level feedback of the run outcome, which task failed which succeeded. Making failures easy to spot is paramount. Then comes the results themselves. Ideally we want to be able to compare to previous runs. I.e. for a given task, we've ran the exact same job on a different commit see if the performance changed. We also want to see the difference between newton and physx for instance. We could select which metric we are interested in comparing to. And we may want to have anither filter based on the machine type. Eventually, we may deploy on different machines (DGX Spark, server, workstations, etc..)
Once we know the system well, we could also put together reference numbers for these different tasks (wrt to reward episode length resource start-up time) and based on this, have a quick overview showing us which tasks pass the thresholds. Letting us see what's up.

## T5: Add the benchmarking harness of IsaacLab 3.0 to IsaacLab 2.3.X

To understand how IsaacLab 3 compares to Isaaclab 2, we'll need to run the old code. But the instrumentation of this code is not as good. So a solution would be to create a branch off IsaacLab 2.3.2 and instrument this code similarly to Isaaclab 3.0. This would allow us to compare "apples to apples" the two frameworks.
