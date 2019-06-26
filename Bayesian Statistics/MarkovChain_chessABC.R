#Transition Matrix
Q = matrix(c(0,1,0.3,0.7),nrow = 2, ncol = 2, byrow = TRUE)


#suppose that the first game is between Players B and C. What is the probability that Player A will play in Game 4?

# is it already known that player A is not playing in round 1.So we need to find Q3
Q3 = Q
for(i in 2:3){
  Q3 = Q3 %*% Q
}

#P(A4| A1=0) = 0.790


# If the players draw from the stationary distribution in Question 4 to decide whether Player A participates in Game 1,
# what is the probability that Player A will participate in Game 4? Round your answer to two decimal places.


# Calculate Q4

# Probability that player A plays in round 4 is given by 1st row column2 of matrix Q*Q*Q*Q
Q4 = Q
for (i in 2:4){
  Q4 = Q4 %*% Q
}  
  # P(A4) = 0.7630

  
  #Calculate Q30
  
Q30 = Q
for (i in 2:30){
  Q30 = Q30 %*% Q
} 

Q30[1,] %*% Q # gives back Q
