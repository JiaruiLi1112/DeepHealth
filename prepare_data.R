library(data.table)
setDTthreads(40)
library(readr)
field_id <- read.csv("field_id.txt", header = FALSE)
uid <- field_id$V1
big_path   <- "/mnt/storage/shared_data/UKBB/20230518-from-zhourong/HHdata_221103_0512.csv"
header_dt  <- fread(big_path, nrows = 0)     # Read 0 rows => only column names
all_names  <- names(header_dt)
keep_names <- intersect(all_names,uid)
ukb_disease <- fread(big_path,
                 select     = keep_names,
                 showProgress = TRUE)

field_id <- read.csv("field_id.txt", header = FALSE)
uid <- field_id$V1
big_path <- "/mnt/storage/shared_data/UKBB/20230518-from-zhourong/HH_data_220812_0512.csv"
header_dt  <- fread(big_path, nrows = 0)     # Read 0 rows => only column names
all_names  <- names(header_dt)
keep_names <- intersect(all_names,uid)
ukb_others <- fread(big_path,
                 select     = keep_names,
                 showProgress = TRUE)

# merge disease and other data by "eid"
ukb_data <- merge(ukb_disease, ukb_others, by = "eid", all = TRUE)
fwrite(ukb_data, "ukb_data.csv")