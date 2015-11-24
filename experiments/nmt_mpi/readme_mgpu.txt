1. The master process is slower than worker process. 
   Probabliy because master process's host is using only one gpu, where as workers' host are using 2 gpus. 

2. When running with 9 gpus, some process will just crash, and this crash will propagate to other GPU. 
   Probably because some GPU return a bad parameter, and other gpu just start from those bad parameters.

