import click, json, csv
from pypairs.tools.sandbag import sandbag, settings, logg
import pandas as pd


@click.command()
@click.option('-d', '--data', required=True,
              type=click.Path(exists=True, file_okay=True, readable=True, resolve_path=True),
              help="Path to the data matrix where rows correspond to cells and columns to genes.")
@click.option('-s', '--sep', default=",", show_default=True, help="Seperator used for data")
@click.option('-t', '--transpose', default=False, is_flag=True, show_default=True, help="Set flag to transpose matrix")
@click.option('-o', '--out', type=click.Path(resolve_path=True, writable=True), help="Store marker pairs to csv")
@click.option('-a', '--annotation', type=click.Path(exists=True, file_okay=True, readable=True, resolve_path=True),
              help='Path to annotation csv. CSV containing at least two columns: Name and Class')
@click.option('-as', '--ann_sep', default=",", help="Seperator used for annotation csv")
@click.option('-an', '--ann_idx_name', default=0, help="Index of name column in annotation csv")
@click.option('-ac', '--ann_idx_class', default=1, help="Index of class column in annotation csv")
@click.option('--ann_header/--ann_no_header', default=True, show_default=True,
              help="Flag weather the annotation contains a header line")
@click.option('-f', '--fraction', default=0.65, show_default=True,
              help="Fraction of cells per category where marker criteria must be satisfied.")
@click.option('-v', '--verbosity', default=3, show_default=True,
              help="Set level of verbosity from 1 to 4. Where 1 is minimal and 4 most")
def main(data, annotation, fraction, out, sep, transpose, verbosity, ann_sep, ann_idx_name, ann_idx_class, ann_header):
    data_mat = pd.read_csv(data, index_col=0, sep=sep)
    if transpose:
        data_mat = data_mat.T

    ann_dict = {}
    with open(annotation) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=ann_sep)
        for i, row in enumerate(readCSV):
            if i == 0 and ann_header:
                continue
            if row[ann_idx_class] not in ann_dict:
                ann_dict[row[ann_idx_class]] = []
            ann_dict[row[ann_idx_class]].append(row[ann_idx_name])

    settings.verbosity = verbosity

    marker_pairs = sandbag(
        data=data_mat, annotation=ann_dict, fraction=fraction,
    )

    if out is not None:
        try:
            with open(out, 'w') as outfile:
                json.dump(marker_pairs, outfile)

            logg.hint("Written marker_pairs to {}".format(out))
        except IOError:
            logg.error("Could not write score to {}".format(out))
    else:
        print(marker_pairs)


if __name__ == "__main__":
    main()
