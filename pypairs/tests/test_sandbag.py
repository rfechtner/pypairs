from pypairs import datasets, settings, utils, pairs
from pandas import DataFrame

settings.verbosity = 4

ref_markers = {
    "G1": [
        ("OGFOD1","ZMYND8"),("TAT","HIST1H3B"), ("APOL4","ANKRD36B"), ("PRR5","HIST1H3B"), ("MFSD4","HIST1H3B"),
        ("PARP3","HIST1H3B"), ("IL10RB","HIST1H3B")
    ],
    "S": [
        ("SNX5","CKS2"),("SLC29A2","GTSE1"),("ZMYND8","CKS2"),("PLEKHJ1","NDUFAF2"),("NSUN5","NDUFAF2"),
        ("CLSPN","CKS2"),("LITAF","CKS2"),("ANKRD36B","METTL11A"),("ANKRD36B","HNRNPUL1"),("ANKRD36B","RPL39L"),
        ("PRELP","CKS2"),("GABRB3","CKS2"),("RBPJ","RPL27"),("TXLNA","RPL27"),("PLEKHJ1","CHMP4A"),("FLNA","TSTD1"),
        ("FLOT1","NEK3"),("BSG","COX6C"),("JKAMP","CKS2"),("TUBB2B","CKS2"),("INSR","CKS2"),("NANOG","CKS2"),
        ("ARIH2","CKS2"),("SEPW1","CKS2"),("SLC7A3","CKS2"),("CHD4","CKS2"),("FTH1","CKS2"),("ZWINT","CKS2"),
        ("ZNF195","CKS2"),("DHX36","CKS2"),("EEF1D","CKS2"),("MCM5","CKS2"),("SLC3A2","CKS2"),("NDUFB8","CKS2"),
        ("FLNA","NDUFA7"),("SLC25A3","SNRPD1"),("RBPJ","RPL31"),("TXLNA","RPL31"),("SLC29A2","CENPL"),("WDR27","BAX"),
        ("SMARCA4","UBLCP1"),("KLRD1","COX17"),("ACBD7","COX17"),("PDDC1","COX17"),("BSG","SF3B14"),("FLNA","COX17")
    ],
    "G2M": [
        ("KPNA2","H2AFZ"),("RPS6","H2AFZ")
    ]}


def test_sandbag_1j():
    settings.n_jobs = 1

    training_data = datasets.leng15(mode='sorted', gene_sub=list(range(0, 1000)))
    marker_pairs = pairs.sandbag(training_data)

    if not utils.same_marker(marker_pairs, ref_markers):
        raise AssertionError()


def test_sandbag_2j_ann():
    settings.n_jobs = 2

    training_data = datasets.leng15(mode='sorted', gene_sub=list(range(0, 1000)))
    annotation = {
        cat: [i for i, x in enumerate(training_data.obs['category']) if x == cat]
        for cat in ["G1", "S", "G2M"]
    }

    marker_pairs = pairs.sandbag(training_data, annotation=annotation)

    if not utils.same_marker(marker_pairs, ref_markers):
        raise AssertionError()


def test_sandbag_unjitted():
    training_data = datasets.leng15(mode='sorted', gene_sub=list(range(0, 1000)))

    marker_pairs_unjitted = pairs.sandbag(training_data)

    if not utils.same_marker(marker_pairs_unjitted, ref_markers):
        raise AssertionError()


def test_sandbag_df():
    training_data = datasets.leng15(mode='sorted', gene_sub=list(range(0, 1000)))
    training_data_df = DataFrame(training_data.X)

    sample_names = list(training_data.obs_names)
    gene_names = list(training_data.var_names)

    training_data_df.Index = sample_names
    training_data_df.columns = gene_names

    annotation = {
        cat: [i for i, x in enumerate(training_data.obs['category']) if x == cat]
        for cat in ["G1", "S", "G2M"]
    }

    marker_pairs_df = pairs.sandbag(training_data_df, annotation)

    if not utils.same_marker(marker_pairs_df, ref_markers):
        raise AssertionError()

    marker_pairs_df = pairs.sandbag(training_data_df, annotation, gene_names, sample_names)

    if not utils.same_marker(marker_pairs_df, ref_markers):
        raise AssertionError()


def test_sandbag_np():
    training_data = datasets.leng15(mode='sorted', gene_sub=list(range(0, 1000)))
    training_data_df = DataFrame(training_data.X)
    training_data_np = training_data_df.values

    sample_names = list(training_data.obs_names)
    gene_names = list(training_data.var_names)
    annotation = {
        cat: [i for i, x in enumerate(training_data.obs['category']) if x == cat]
        for cat in ["G1", "S", "G2M"]
    }

    marker_pairs_np = pairs.sandbag(training_data_np, annotation, gene_names, sample_names)

    if not utils.same_marker(marker_pairs_np, ref_markers):
        raise AssertionError()


def test_sandbag_filtered():
    training_data = datasets.leng15(mode='sorted', gene_sub=list(range(0, 1000)))
    sample_names = list(training_data.obs_names)
    sample_names.pop(3)

    marker_pairs_filtered = pairs.sandbag(
        training_data, filter_genes=list(range(0, 999)), filter_samples=sample_names
    )

    ref2 = ref_markers
    ref2['G2M'].append(('CENPL', 'APOL4'))

    if not utils.same_marker(marker_pairs_filtered, ref2):
        raise AssertionError()
