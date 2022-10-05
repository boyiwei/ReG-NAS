# ReG-NAS

## 1 Required Environment

```yaml
cuda version: 11.1
python: 3.7
torch: 1.11.0
pyg (for conda) /torch_geometric (for pip): 2.0.4
tensorboardX: 2.8.0
```
To check whether your environment match the requirement, please run
````shell
bash run_single.sh
````
If no errors occurs, then you can use ReG-NAS. ReG-NAS itself contains ``torch_geometric`` and ``torch.nn`` for we modified
some file in it. However, you still need to install ``torch_geometric`` and  ``torch`` in your environment
## 2. How to use
### 2.1 To run a single experiments

RegNAS offers several training pipelines for experiments. As listed below:
```text
main.py: Groundtruth training pipeline for ogbg-molhiv or other classification based groundturth task.
main_ZINC.py: Groundtruth training pipeline for ZINC or other regression based groundtruth task.
main_proxy_le.py: Proxy training pipeline using Laplacian Matrix's eigenvectors as proxy task.
main_proxy_pm.py: Proxy training pipeline using Poorest Model generated vectors as proxy task.
main_proxy_rm.py: Proxy training pipeline using Randomly-Selected Model generated vectors as proxy task. (also applicable for GM based pipeline)
main_proxy_rv.py: Proxy training pipeline using Random generated vectors as proxy task
```
For example, if we want to run ``main.py`` for 3 times with the configuration ``your_configs.yaml`` (stored in configs/pyg), simply use
```shell
python main.py --cfg configs/your_configs.yaml --repeat 3 # graph classification
```
The result will be stored in ``results/``, where you can analyze the training results


### 2.2 To run experiments in a search space

First you can define your search space in ``grids/``, in which we offered some example files in it.

After defined you search space, the shell files listed below are designed for managing experiments in a search space:
```text
run_batch.sh: Classification-based groundtruth training
run_batch_ZINC.sh Regression-based groundtruth training
run_batch_proxy.sh: Proxy training
```
In these file, you can define your training configurations (hyper-parameter settings) in ``CONFIG``, specify your search space file in ``GRID``, define training pipeline in ``MAIN``, 
and define max parallel jobs in ``MAX_JOBS''. Then simply use
```shell
bash run_batch.sh
```
to run the experiment.

### 2.3 Results analysis

All the functions for result analysis is in ``proxy_groundtruth_analysis.py``, which contains:
```txt
proxy_groundtruth_analysis: Analyzes proxy ranking and classification based groundtruth ranking. Computes rho and tau, draws scatter figure.
proxy_groundtruth_analysis_ZINC: Analyzes proxy ranking and  regression-based grdountruth ranking. Computes rho and tau, draws scatter figure.
proxy_groundtruth_analysis_epoch: Analyzes proxy ranking and classification based groundtruth ranking at each epoch. Computes rho and tau, draws scatter figure.
proxy_groundtruth_analysis_ZINC: Analyzes proxy ranking and regression-based grdountruth ranking at each epoch. Computes rho and tau, draws scatter figure.
similarity_analysis: Analyzes Ranking Stability for groundtruth ranking or proxy ranking. Computes rho and tau, draws scatter figure.
similarity_analysis_epoch: Analyzes Ranking Stability at each epoch. Computes rho and tau, draws scatter figure.
```
Simply add these function below ``if __name__ == '__main__':`` when you want to analyze experiment results. Before using, please change the file path contained in the function.