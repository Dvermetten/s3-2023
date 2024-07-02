## Logging 

To add logging to these problems, we add a simple csv-based logger. This is connected to the model, and any time the model creates a complete solution it should pass this to the logger. The logger only writes to file at the end of a run, and since it is a basic csv-structure the results of multiple runs can be concatenated into a single experimental result. This can then e.g. be loaded into IOHanalyzer using the 'custom csv' upload option. 

### Usage: 
For the tsp-example, simply add the 'performance_file' argument to determine where the log is written. If this is not passed, the code functions as before. 

For the base code, the logger object is included, and added to init and empty solution generation. The calling of this function should be handled based on the problem (whenever a complete solution is created)

### Challenges / Improvements
Currently, this logging only executes on a single run, we could add some wrapping to perform multiple runs / full benchmark exectutions for convenience. The logging is also not very flexible right now, but this can be modified if needed. 

## Irace

The irace interface has been added to both the tsp-example and the base code. The parameter structure for irace is rigid, so I have separated this from the logged / original versions. Running irace can be done via R as follows:
```
library(irace)
p <- readParameters()
s <- readScenario()
irace(s,p)
```

Note that the budgets are currently treated as fixed parameters, so to change budget you need to change the parameter file (or handle it in the R-session). The parameter structure and bounds are currently just a guess, can be further improved