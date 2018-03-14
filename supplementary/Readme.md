
![Test](./images/general/logos_combined.png)

# Maschine learning methods for predicting cell-cycle phase from scRNA-Seq data

## By Ron Fechtner 
### Ludwig-Maximilians-Universität München / Technische Universität München


This is the supplementary for the Bachelor's Thesis on Maschine learning methods for predicting cell-cycle phase from scRNA-Seq data.

Here you will find the a digital copy of the thesis, the codes, data, images, restul tables and juypter notebooks. Most of the chapters have a associated jupyter notebook file within the notebook folder that shows how the results were generated. This Readme is an example of such an notebook. All code can be directly executed or modified. For further information please see: http://jupyter.readthedocs.io/en/latest/index.html

If you don't want to install juypter notebook on your local machine you can visit http://nbviewer.jupyter.org/github/rfechtner/pypairs/tree/master/supplementary/notebooks/ for a online view (read only).

The general structure of this folder is:


```python
import os

for root, dirs, files in os.walk("./"):
    level = root.replace("./", '').count(os.sep)
    indent = ' ' * 4 * (level)
    print()
    folder = os.path.basename(root)
    if not folder.startswith("."):
        print('{}{}/'.format(indent, folder))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
```

    
    /
        Readme.ipynb
        Readme.md
    
    
    code/
        helper.py
        pypairs.py
        __init__.py
    
        __pycache__/
            helper.cpython-36.pyc
            pypairs.cpython-36.pyc
    
    data/
        biomart_human-genes.txt
        biomart_mouse-human-orthologs.txt
        cell_cycle_genes.csv
        cyclebase_top1000_genes.tsv
        E-MTAB-3929.processed.1_counts.txt
        E-MTAB-3929_annotation.txt
        E-MTAB-6142_human.csv
        go_0007049_homoSapiens.csv
        GSE53481_humanRNAseq.txt
        GSE64016_H1andFUCCI_normalized_EC_human.csv
        GSE71456_Samples_RPKM.csv
        GSE75748_bulk_cell_type_ec.csv
        GSE75748_bulk_time_course_ec.csv
        GSE75748_sc_cell_type_ec.csv
        GSE75748_sc_time_course_ec.csv
        mESC_dataset_mouse.txt
        mouse_pretrained-pairs.json
        Non_norm.PolyA_NamedByAlex_human.csv
        regev_lab_cell_cycle_genes.txt
    
    images/
    
        application/
            cell_lineage_e5-e7.pdf
            ebv_circle.pdf
            ebv_circle.png
            ebv_line.pdf
            ebv_line.png
            ebv_scatter.pdf
            prediction_e3-e7.pdf
            prediction_e3-e7.png
            prediction_e3-ee5.pdf
            prediction_e3-ee5.png
            prediction_e5-e7.pdf
            prediction_e5-e7.png
    
        differences/
            sandbag-speed.pdf
            sandbag-speed.png
    
        evaluation/
            E-MTAB-3929.sdrf.txt
            hESC-scatter.pdf
            hESC-scatter.png
            hESC-scatter.svg
            hPSC-assign-all-scatter.pdf
            hPSC-assign-all.pdf
            hPSC-assign-all.png
            hPSC-assign-h1-scatter.pdf
            hPSC-assign-h1.pdf
            hPSC-assign-h1.png
            ml_eval.pdf
            ml_pred.pdf
            mouse_on_human.pdf
            mouse_on_human.png
            mouse_on_human_norm.pdf
            mouse_on_human_norm.png
            oscope-fraction-test-cc-only.pdf
            oscope-fraction-test-cc-only.png
            oscope-fraction-test.pdf
            oscope-fraction-test.png
            oscope-fraction-test_old1.png
            oscope-fraction-test_old2.png
            oscope-pca.pdf
            oscope-pca.png
            oscope-phase-distribution.pdf
            oscope-phase-distribution.png
            prediction-mESC-on-hESC.png
    
        extension/
            networkx.pdf
            networkx.png
            pairs_ematb6142.pdf
            pairs_ematb6142.png
            rfp_ematb6142.pdf
            rfp_ematb6142.png
            rf_bulk.pdf
            rf_bulk.png
            triplets.pdf
    
        general/
            logo1.jpg
            logo2.jpg
            logos_combined.PNG
    
    notebooks/
    
        2. Python reimplementation/
            2.3 Differences in code - Python.ipynb
            2.3 Differences in code - R.ipynb
    
        3. Evaluation/
            3.2 Mouse pairs on human dataset.ipynb
            3.3 Internal cross validation.ipynb
            3.4.1.1 Bulk - GSE53481.ipynb
            3.4.1.2 Bulk - GSE71456.ipynb
            3.4.2 Single cell - EMATB6142.ipynb
    
        4. Application/
            4.1 EBV.ipynb
            4.2 E-MTAB-3929.ipynb
    
        5. Extension/
            5.1 Random forest on pairs.ipynb
            5.2 Pairs network.ipynb
            5.2.1 Weighted Pairs.ipynb
            5.2.2. Weighted Triplets.ipynb
    

### The PyPairs implementation is also available via pip:
```shell
$ pip install pypairs
``` 

### And Github:
https://github.com/rfechtner/pypairs

### For questions please write me at:
ronfechtner@gmail.com
