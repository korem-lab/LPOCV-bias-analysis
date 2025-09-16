

library(tidyverse)
library(data.table)
load('taxa_data.Rdata') ## available from 
                  ## https://github.com/hczdavid/metaManuscript/tree/main/Analyses/Data

data.table::rbindlist( taxa_data$prop$com_V1 %>% 
                         lapply(function(x) return(as.data.frame(x$ASV, 
                                                                 row.names=row.names(x))) ), 
                       use.names=TRUE ) %>%
  write.csv('../../data/cal_ptb/ptb_data.csv')



data.table::rbindlist( lapply( names(taxa_data$prop$com_V1), 
                               function(x) return(taxa_data$meta[[x]] ) ),
                       use.names = T )%>%
  write.csv('../../data/cal_ptb/ptb_metadata.csv')
