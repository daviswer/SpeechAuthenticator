%values are given in order from 2-class to 9-class
expected_error = [ 1/2, 2/3, 3/4, 4/5, 5/6, 6/7, 7/8, 8/9 ]
train_accuracy = [ 
    0.965 , 
    0.805  ,
    0.628  ,
    0.513  ,
    0.375   ,
    0.334   ,
    0.199  , 
    0.173   
    ]
test_accuracy = [
    0.804, 
    0.588, 
    0.466, 
    0.394, 
    0.301, 
    0.268, 
    0.161,
    0.148
    ]
f1_score = [
    0.804, 
    0.586, 
    0.461, 
    0.390, 
    0.325, 
    0.264, 
    0.197, 
    0.173
    ]  
improvement = 1.-(1.-test_accuracy)'./expected_error  
    
improvement = [
    0.608,
    0.382,
    0.288,
    0.243,
    0.161,
    0.146,
    0.041,
    0.042
];