from pypairs import pairs
from pypairs import datasets

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

def same_marker(a,b):
    if len(a) != len(b):
        return False

    if sorted(a.keys()) != sorted(b.keys()):
        return False

    for cat, values in a.items():
        set_a = set(values)
        set_b = set(b[cat])

        if set_a - set_b or set_b - set_a:
            return False

    return True


def test_sandbag():
    training_data = datasets.leng15(mode='sorted', gene_sub=list(range(0,1000)))
    marker_pairs = pairs.sandbag(training_data)

    assert same_marker(marker_pairs,ref_markers)
