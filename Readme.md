# This is the code for paper: Exploiting Multifaceted Nature of Items and Users for Session-based Recommendation

## Requirement

- Python - 3.8.10
- pytorch - 1.13.1+cu117
- numpy - 1.24.3
- entmax - 1.1



## Run Code

Just execute `python main.py` to run the script. You can specify datasets like yoo64, RetailRocket, and Tmall by setting the `--dataset` option.



## Tips

- Here, we provide the dataset processed from the original sequence, which includes -neighbor, -session, -train, and -test. 
- For the Tmall dataset, we found that using the original Item Embedding yields better results when calculating the loss (formula 4.16), therefore, we have set the Tmall dataset as a separate new .py file.