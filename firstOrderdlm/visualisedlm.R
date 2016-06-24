## Check and install packages
packages <- c("dplyr","tidyr","ggplot2","gridExtra", "magrittr")
newPackages <- packages[!(packages %in% as.character(installed.packages()[,"Package"]))]
if(length(newPackages)) install.packages(newPackages)
lapply(packages,require,character.only=T)

## set ggplot theme
theme_set(theme_minimal())

## read in the data and set column names
data = read.csv("firstOrderdlm.csv")
colnames(data) = c("stationId", "time", "observation", "state")

## clean the data using dplyr and plot using ggplot2
data %>%
  gather(key = "key", value = "value", -time, -stationId) %>%
  filter(stationId %in% 1:4) %>%
  ggplot(aes(x = time, y = value, colour = key)) + 
  geom_line() +
  facet_wrap(~stationId, scales = "free")

##################################
# Visualise Kalman Filtered Data #
##################################

filtered = read.csv("filteredDlm.csv")
colnames(filtered) = c("stationId", "time", "observation", "stateEstimate", "v", "w", "m", "c")

## calculate upper and lower 95% bounds of the state estimate
filtered %<>%
  mutate(upper = qnorm(p = 0.975, mean = m, sd = sqrt(c)), 
         lower = qnorm(p = 0.025, mean = m, sd = sqrt(c)))

## join to the original data to see how close the Kalman Filter estimates the state
filtered %<>%
  inner_join(data[,-3], by = c("time", "stationId")) %>%
  select(-v, -w, -m, -c, -observation) %>%
  gather(key = "key", value = "value", -time, -stationId)

## Plot state estimate and intervals
filtered %>%
  filter(stationId %in% 1:4) %>%
  ggplot(aes(x = time, y = value, colour = key, linetype = key)) + 
  geom_line() +
  facet_wrap(~stationId, scales = "free")
