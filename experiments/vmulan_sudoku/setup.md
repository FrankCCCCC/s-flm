# Goal: Implement Learnable Noise Scheudler

Refer to ``TODO:`` tags in the codebase. I want to build a learnable noise scheduler parametrized by NN. Given it can offer global or positional-wise time schedule for EFLM. Positional-wise time schedule means it will offer time schedulers for each position to achieve learnable decoding order. 

After implementation, train models with default hyperparameter on Sudoku hard with RHO: {5.0, 8.0, 16.0}

---