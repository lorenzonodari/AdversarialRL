#!/bin/bash

# Retrieve trained CW agents + results
docker cp TrainPerfectCW:/code/results ./cw_results
docker cp TrainPerfectCW:/code/agents ./cw_agents
mv cw_results/* ./results
mv cw_agents/* ./agents
rmdir cw_results
rmdir cw_agents

# Retrieve trained KW agents + results
docker cp TrainPerfectKW:/code/results ./kw_results
docker cp TrainPerfectKW:/code/agents ./kw_agents
mv kw_results/* ./results
mv kw_agents/* ./agents
rmdir kw_results
rmdir kw_agents

# Retrieve trained SW agents + results
docker cp TrainPerfectSW:/code/results ./sw_results
docker cp TrainPerfectSW:/code/agents ./sw_agents
mv sw_results/* ./results
mv sw_agents/* ./agents
rmdir sw_results
rmdir sw_agents
