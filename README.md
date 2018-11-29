# Explaining Knowledge base completion

We propose a method, that can be used to explain why an embedding based neural model trained for knowledge base completion gave that specific answer for a query. More specifically, given a knowledge base represented as a list of triples (e<sub>1</sub>, r, e<sub>2</sub>), we train an embedding based neural model to answer the query (e<sub>1</sub>, r, ?) which give the answer as u.

We try to give reasoning for that answer for an arbitrary embedding based neural model. Presently, we are using the state of the art model [TypeDM](https://github.com/dair-iitd/KBI/tree/master/kbi-pytorch)


## Running Instructions

### Requirements

```
python >= 3.6
pytorch (version 0.3.1.post2)
numpy (version 1.13.3)
sklearn (version 0.19.1)
matplotlib (version 2.1.0)
pickle
```

### Template Builder

We have a set of fixed templates, which we suppose are enough to explain some part of the dataset. We need to build template tables for these templates.
To do so, run the `template_builder.py` file

```
python3 template_builder.py -h       ## Get help
python3 template_builder.py -d fb15k -m distmult -w dumps/fb15k_distmult_dump_norm.pkl -s logs/fb15k -v 1 --t_ids 1 2 3 4 5 --data_repo_root ../data
```

This will save `1-5.pkl` in the save_directory.

### Preprocessing

We need to preprocess the textual data to numeric data for our selection module as an input. To do so run the file `preprocessing.py` as given below:

```
python3 preprocessing.py -d fb15k -m distmult -f ../data/fb15k/train.txt -s logs/fb15k/sm_with_id.data -w dumps/fb15k_distmult_dump_norm.pkl -l logs/fb15k -v 1 --t_ids 1 2 3 4 5 --data_repo_root ../data --negative_count 2
```

### Training Selection Module
Next to train the selection module run the file `sm/main.py` as given below:
```
python3 main.py --training_data_path ../logs/fb15k-inv/sm_with_id.data.pkl --base_model_file ../dumps/fb15k_inv_distmult_dump_norm.pkl --val_data_path ../logs/fb15k-inv/exp_words/sm_valid_with_id.pkl --exp_name test --output_path logs/ --num_epochs 100 --log_after 200000 --batch_size 2048 --use_ids --mil --each_input_size 7 --num_templates 5 --config config_id.yml --cuda
```

## Authors

* **Aman Agrawal** - [http://www.cse.iitd.ernet.in/~cs1150210/](http://www.cse.iitd.ernet.in/~cs1150210/)
* **Ankesh Gupta** - [https://www.linkedin.com/in/ankesh-gupta-a67423123](https://www.linkedin.com/in/ankesh-gupta-a67423123)
* **Yatin Nandwani** - [https://www.linkedin.com/in/yatin-nandwani-0804ba9/](https://www.linkedin.com/in/yatin-nandwani-0804ba9/)

See also the list of [contributors](https://github.com/aman71197/Interpretable-KBC/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Project completed under the guidance of

* **Mausam** - [http://www.cse.iitd.ac.in/~mausam/](http://www.cse.iitd.ac.in/~mausam/)
* **Parag Singla** - [http://www.cse.iitd.ac.in/~parags/](http://www.cse.iitd.ac.in/~parags/)