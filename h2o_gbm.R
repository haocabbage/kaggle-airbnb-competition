library(h2o)
h2o.init(nthreads = -1, max_mem_size = "40G")
h2o.removeAll()

gbm.temp <- h2o.gbm(training_frame = tr.temp,     
                    validation_frame = va.temp,   
                    x= 2:610,                     
                    y= 1,                        
                    ntrees = 50,                
                    learn_rate = 0.2,           
                    max_depth = 20,              
                    sample_rate = 0.7,          
                    col_sample_rate = 0.7,       
                    stopping_rounds = 2,         
                    stopping_tolerance = 0.01,  
                    score_each_iteration = T,   
                    seed = 42)             

gbm.model <- h2o.gbm(training_frame = X.h2o,     
                    x= 2:610,                     
                    y= 1,                        
                    ntrees = 50,                
                    learn_rate = 0.2,           
                    max_depth = 20,              
                    sample_rate = 0.7,          
                    col_sample_rate = 0.7,       
                    stopping_rounds = 2,         
                    stopping_tolerance = 0.01,  
                    score_each_iteration = T,   
                    seed = 42)  

tic <- Sys.time()
gbm.data.model <- h2o.gbm(training_frame = data.tr.h2o,     
                          x= 2:314,                     
                          y= 1,                        
                          ntrees = 50,                
                          learn_rate = 0.1,           
                          max_depth = 20,              
                          sample_rate = 0.7,          
                          col_sample_rate = 0.7,       
                          stopping_rounds = 2,         
                          stopping_tolerance = 0.01,  
                          score_each_iteration = T,   
                          seed = 42) 
Sys.time() - tic

tic <- Sys.time()
gbm.data.log.model <- h2o.gbm(training_frame = data.log.tr.h2o,     
                              x= 2:314,                     
                              y= 1,                        
                              ntrees = 50,                
                              learn_rate = 0.1,           
                              max_depth = 20,              
                              sample_rate = 0.7,          
                              col_sample_rate = 0.7,       
                              stopping_rounds = 2,         
                              stopping_tolerance = 0.01,  
                              score_each_iteration = T,   
                              seed = 42) 
Sys.time() - tic

h2o.shutdown()