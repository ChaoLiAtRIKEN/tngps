tnGPS: Discovering Unknown Tensor Network Structure Search Algorithms via Large Language Models (LLMs) (ICML, 2024) [https://proceedings.mlr.press/v235/zeng24b.html]
===================================

Introduction
-------------------------------
This repository is the implementation of tnGPS.



Requirements
----------------------
 * Python 3.7.3<br/>
 
 * Tensorflow 1.13.1
 
Usage
---------------------
First, you need to start agents with

     CUDA_VISIBLE_DEVICES=0 python agent.py 0
     
The last 0 stands for the id of the agent. You can spawn multiple agents with each one using one GPU by modifying the visible device id. <br/>

Then start the main script by

     python tnGPS.py 100
     
The argv stands for the number of samples of the TN-SS algorithms in one iteration.

