expected_error = [ 1/2, 2/3, 3/4, 4/5, 5/6, 6/7, 7/8, 8/9 ]
train_accuracy = [ 
    0.883458646617, 
    0.742816500711, 
    0.669039145907, 
    0.630640941792, 
    0.592393495498, 
    0.562994652406, 
    0.525318795047, 
    0.478151804499
    ]
test_accuracy = [
    0.874316939891, 
    0.730947647449, 
    0.666178623719, 
    0.615179252479, 
    0.556878721404, 
    0.522704590818, 
    0.489003880983, 
    0.445828614572
    ]
f1_score = [
    0.874420950896, 
    0.731517308737, 
    0.666579117388, 
    0.615398724056, 
    0.555470149322, 
    0.470622788513, 
    0.485977114985, 
    0.443577893463
    ]  
improvement = 1.-(1.-test_accuracy)'./expected_error    
    
improvement = [
    0.74863,
    0.59642,
    0.55490,
    0.51897,
    0.46825,
    0.44316,
    0.41600,
    0.37656]