# script.R
df <- data.frame(x=1:10, y=(1:10)+rnorm(n=10))
print(df)
linModel <- lm(y ~ x, df)
print(summary(linModel))
